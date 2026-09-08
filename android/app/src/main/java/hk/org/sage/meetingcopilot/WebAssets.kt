package hk.org.sage.meetingcopilot

import android.content.Context
import java.io.File

/**
 * Copies `assets/web/**` (templates/ and static/, staged by Gradle from the
 * repository) to the app's private files directory, where Flask can read them
 * as ordinary files. Assets inside an APK are not files, and Flask has no
 * notion of anything else.
 */
object WebAssets {
    private const val ROOT = "web"

    fun unpack(context: Context): File {
        val target = context.filesDir.resolve(ROOT)
        // Start clean so a file removed from the repository does not linger.
        target.deleteRecursively()
        copyTree(context, ROOT, target)
        return target
    }

    private fun copyTree(context: Context, assetPath: String, into: File) {
        val assets = context.assets
        val children = assets.list(assetPath) ?: emptyArray()
        if (children.isEmpty()) {
            // AssetManager.list() is empty for a file (and for an empty
            // directory, which we never ship), so this is a file: copy it.
            into.parentFile?.mkdirs()
            assets.open(assetPath).use { input ->
                into.outputStream().use { output -> input.copyTo(output) }
            }
            return
        }
        into.mkdirs()
        for (child in children) {
            copyTree(context, "$assetPath/$child", into.resolve(child))
        }
    }
}
