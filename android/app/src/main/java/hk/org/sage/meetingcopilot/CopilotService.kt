package hk.org.sage.meetingcopilot

import android.Manifest
import android.app.Notification
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlin.system.exitProcess

/**
 * Runs the Python server as a foreground service.
 *
 * Android kills background work aggressively; a foreground service with a
 * visible notification is the one sanctioned way to keep a two-hour meeting
 * alive when the screen turns off. The Python thread itself is started exactly
 * once per process -- Flask's development server cannot be stopped and
 * restarted, so "Quit" ends the process rather than pretending to.
 */
class CopilotService : Service() {

    companion object {
        const val CHANNEL_ID = "copilot"
        const val NOTIFICATION_ID = 1
        const val PORT = 5000
        const val ACTION_QUIT = "hk.org.sage.meetingcopilot.QUIT"
        private const val TAG = "CopilotService"

        private val lock = Any()
        @Volatile private var serverThreadStarted = false

        fun start(context: Context) {
            context.startForegroundService(Intent(context, CopilotService::class.java))
        }
    }

    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        // A partial wake lock keeps the CPU running with the screen off. Ten
        // hours is longer than any meeting and shorter than "forever", which
        // lint rightly complains about.
        val power = getSystemService(PowerManager::class.java)
        wakeLock = power.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "MeetingCopilot::server").apply {
            acquire(10 * 60 * 60 * 1000L)
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_QUIT) {
            Log.i(TAG, "quit requested")
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            // The only way to actually stop the embedded server.
            exitProcess(0)
        }

        // The activity calls start() on launch and again once the microphone
        // permission is decided, so the service type can be upgraded to
        // "microphone" -- which Android 15 exempts from the 6-hour dataSync cap.
        val micGranted = checkSelfPermission(Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        val type = when {
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && micGranted ->
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
            else -> ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
        }
        startForeground(NOTIFICATION_ID, buildNotification(), type)

        startServerOnce()
        return START_STICKY
    }

    override fun onDestroy() {
        wakeLock?.let { if (it.isHeld) it.release() }
        super.onDestroy()
    }

    private fun startServerOnce() {
        synchronized(lock) {
            if (serverThreadStarted) return
            serverThreadStarted = true
        }
        val appContext = applicationContext
        Thread({
            try {
                if (!Python.isStarted()) {
                    Python.start(AndroidPlatform(appContext))
                }
                // The web files travel as assets; Flask needs them as real
                // files on disk, so they are unpacked on every start (they
                // are small, and this way an update can never leave a stale copy).
                val web = WebAssets.unpack(appContext)
                val data = appContext.filesDir.resolve("data")
                Log.i(TAG, "starting python server on 127.0.0.1:$PORT, data in $data")
                Python.getInstance()
                    .getModule("android_entry")
                    .callAttr(
                        "start",
                        data.absolutePath,
                        web.resolve("templates").absolutePath,
                        web.resolve("static").absolutePath,
                        PORT,
                    )
                // app.run() blocks for the life of the process; reaching here
                // means it returned, which it only does on failure.
                Log.e(TAG, "python server exited")
            } catch (e: Throwable) {
                Log.e(TAG, "python server failed", e)
            }
        }, "python-server").start()
    }

    private fun buildNotification(): Notification {
        val open = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val quit = PendingIntent.getService(
            this, 1,
            Intent(this, CopilotService::class.java).setAction(ACTION_QUIT),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(getString(R.string.notification_title))
            .setContentText(getString(R.string.notification_text))
            .setContentIntent(open)
            .addAction(0, getString(R.string.notification_quit), quit)
            .setOngoing(true)
            .setSilent(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }
}
