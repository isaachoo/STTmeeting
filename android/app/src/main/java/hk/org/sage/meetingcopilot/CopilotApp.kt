package hk.org.sage.meetingcopilot

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager

class CopilotApp : Application() {
    override fun onCreate() {
        super.onCreate()
        // The foreground service needs a channel to post its notification on.
        // Created here so it exists before the first startForeground() call.
        val channel = NotificationChannel(
            CopilotService.CHANNEL_ID,
            getString(R.string.notification_channel),
            NotificationManager.IMPORTANCE_LOW,  // no sound, no heads-up
        ).apply {
            description = getString(R.string.notification_text)
            setShowBadge(false)
        }
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }
}
