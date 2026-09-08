package hk.org.sage.meetingcopilot

import android.content.ContentValues
import android.content.Context
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.webkit.URLUtil
import android.widget.Toast
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/**
 * The web page offers downloads (Markdown/JSON exports, reports) with a
 * Content-Disposition header. A WebView ignores those unless told what to do,
 * so this fetches the file from the local server and drops it in the phone's
 * Downloads folder through MediaStore -- no storage permission needed on
 * Android 10+, and it shows up in Files and in every "attach a file" picker.
 */
object Downloads {
    fun save(context: Context, url: String, contentDisposition: String?, mimeType: String?) {
        val app = context.applicationContext
        Thread({
            val name = URLUtil.guessFileName(url, contentDisposition, mimeType)
            try {
                val connection = URL(url).openConnection() as HttpURLConnection
                connection.connectTimeout = 10_000
                connection.readTimeout = 60_000
                if (connection.responseCode != 200) {
                    throw IOException("HTTP ${connection.responseCode}")
                }
                val resolver = app.contentResolver
                val values = ContentValues().apply {
                    put(MediaStore.Downloads.DISPLAY_NAME, name)
                    put(MediaStore.Downloads.MIME_TYPE, mimeType?.ifBlank { null } ?: "application/octet-stream")
                    put(MediaStore.Downloads.IS_PENDING, 1)
                }
                val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                    ?: throw IOException("MediaStore refused the file")
                connection.inputStream.use { input ->
                    resolver.openOutputStream(uri)!!.use { output -> input.copyTo(output) }
                }
                values.clear()
                values.put(MediaStore.Downloads.IS_PENDING, 0)
                resolver.update(uri, values, null, null)
                toast(app, app.getString(R.string.saved_to_downloads, name))
            } catch (e: Exception) {
                toast(app, app.getString(R.string.save_failed, e.message ?: e.javaClass.simpleName))
            }
        }, "download").start()
    }

    private fun toast(context: Context, message: String) {
        Handler(Looper.getMainLooper()).post {
            Toast.makeText(context, message, Toast.LENGTH_LONG).show()
        }
    }
}
