// The Android wrapper for the meeting copilot. Open THIS folder in Android
// Studio, not the repository root. See ../ANDROID.md for the full walkthrough.
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "MeetingCopilot"
include(":app")
