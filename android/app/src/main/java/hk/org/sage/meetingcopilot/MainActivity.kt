package hk.org.sage.meetingcopilot

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.View
import android.view.WindowManager
import android.webkit.PermissionRequest
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.TextView
import android.widget.Toast
import androidx.activity.addCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.net.InetSocketAddress
import java.net.Socket

/**
 * The whole UI is the same web page a PC shows, in a WebView pointed at the
 * server running inside this app. This activity only has to: get the
 * microphone permission, keep the server alive, wait until it answers, and
 * hand downloads and external links to the system.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var web: WebView
    private lateinit var statusPanel: View
    private lateinit var status: TextView

    /** A getUserMedia() request the page made before the OS permission existed. */
    private var pendingWebPermission: PermissionRequest? = null
    private var pageLoaded = false

    private val askPermissions = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { grants ->
        // Now that the answer is known, (re)start the service so it can pick
        // the "microphone" foreground type instead of "dataSync".
        CopilotService.start(this)
        val micOk = grants[Manifest.permission.RECORD_AUDIO] == true
        pendingWebPermission?.let { request ->
            if (micOk) request.grant(request.resources) else request.deny()
        }
        pendingWebPermission = null
        if (!micOk) Toast.makeText(this, R.string.mic_needed, Toast.LENGTH_LONG).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        web = findViewById(R.id.web)
        statusPanel = findViewById(R.id.status_panel)
        status = findViewById(R.id.status)

        // A phone on the meeting table should not dim and lock halfway through
        // -- and a WebView whose activity is stopped may pause microphone
        // capture, so the safe rule is: while this screen is open, it stays on.
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // Start the server straight away (dataSync type until the mic permission
        // is known), then ask for permissions; the callback restarts it.
        CopilotService.start(this)
        requestPermissionsIfNeeded()

        configureWebView()
        onBackPressedDispatcher.addCallback(this) {
            if (web.canGoBack()) web.goBack() else moveTaskToBack(true)
        }

        waitForServerThenLoad()
    }

    // ------------------------------------------------------------ permissions

    private fun hasMic(): Boolean =
        checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

    private fun requestPermissionsIfNeeded() {
        val wanted = mutableListOf<String>()
        if (!hasMic()) wanted += Manifest.permission.RECORD_AUDIO
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            wanted += Manifest.permission.POST_NOTIFICATIONS
        }
        if (wanted.isNotEmpty()) askPermissions.launch(wanted.toTypedArray())
    }

    // --------------------------------------------------------------- webview

    private fun configureWebView() {
        with(web.settings) {
            javaScriptEnabled = true
            domStorageEnabled = true
            mediaPlaybackRequiresUserGesture = false  // speak-aloud TTS
            allowFileAccess = false
            allowContentAccess = false
        }
        if (BuildConfig.DEBUG) {
            // chrome://inspect on a PC connected by USB shows the page's console.
            WebView.setWebContentsDebuggingEnabled(true)
        }

        web.webChromeClient = object : WebChromeClient() {
            override fun onPermissionRequest(request: PermissionRequest) {
                // The page asked for the microphone (getUserMedia). Grant it only
                // if the OS permission exists; otherwise ask and answer later.
                val wantsAudio = request.resources.contains(PermissionRequest.RESOURCE_AUDIO_CAPTURE)
                when {
                    !wantsAudio -> request.deny()
                    hasMic() -> request.grant(request.resources)
                    else -> {
                        pendingWebPermission = request
                        askPermissions.launch(arrayOf(Manifest.permission.RECORD_AUDIO))
                    }
                }
            }
        }

        web.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                val url = request.url
                // Everything on the local server stays in the app; anything else
                // (a web-search source the copilot cited) opens in the browser.
                if (url.host == "127.0.0.1" || url.host == "localhost") return false
                startActivity(Intent(Intent.ACTION_VIEW, url))
                return true
            }

            override fun onPageFinished(view: WebView, url: String) {
                if (!pageLoaded) {
                    pageLoaded = true
                    statusPanel.visibility = View.GONE
                    web.visibility = View.VISIBLE
                }
            }

            override fun onReceivedError(view: WebView, request: WebResourceRequest, error: WebResourceError) {
                if (!request.isForMainFrame) return
                // The server was up a moment ago and is not now: wait and retry
                // rather than showing Chromium's grey error page.
                pageLoaded = false
                web.visibility = View.INVISIBLE
                statusPanel.visibility = View.VISIBLE
                status.setText(R.string.starting)
                waitForServerThenLoad()
            }
        }

        web.setDownloadListener { url, _, contentDisposition, mimeType, _ ->
            Downloads.save(this, url, contentDisposition, mimeType)
        }
    }

    /**
     * The Python server takes a few seconds to come up (longer on the very
     * first launch, while Chaquopy unpacks the standard library). Poll the
     * port from a background thread and load the page the moment it answers.
     */
    private fun waitForServerThenLoad() {
        Thread({
            val deadline = System.currentTimeMillis() + 120_000
            var attempt = 0
            var up = false
            while (System.currentTimeMillis() < deadline) {
                if (portOpen()) { up = true; break }
                attempt++
                if (attempt == 20) runOnUiThread { status.setText(R.string.still_starting) }
                Thread.sleep(500)
            }
            runOnUiThread {
                if (up) {
                    web.loadUrl("http://127.0.0.1:${CopilotService.PORT}/")
                } else {
                    status.setText(R.string.server_failed)
                    findViewById<View>(R.id.spinner).visibility = View.GONE
                }
            }
        }, "wait-for-server").start()
    }

    private fun portOpen(): Boolean = try {
        Socket().use { it.connect(InetSocketAddress("127.0.0.1", CopilotService.PORT), 400) }
        true
    } catch (e: Exception) {
        false
    }

    override fun onDestroy() {
        // The activity may be recreated (rotation, theme change); the server
        // lives in the service and must not be torn down with the WebView.
        web.destroy()
        super.onDestroy()
    }
}
