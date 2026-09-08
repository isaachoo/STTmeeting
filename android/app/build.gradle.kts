// The app is a thin shell: the same Python server that runs on a PC, started
// inside the app by Chaquopy, with a WebView pointed at http://127.0.0.1:5000.
// Nothing about the meeting copilot itself is rewritten -- the Python code and
// the web files are copied in from the repository at build time.

import java.io.File

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

// This module lives at <repo>/android/app, so the copilot's code is two up.
val repoRoot: File = rootProject.projectDir.parentFile
val stagedPython = layout.buildDirectory.dir("staged/python")
val stagedAssets = layout.buildDirectory.dir("staged/assets")

// -- staging -------------------------------------------------------------
// Chaquopy packages whatever is in its Python source directory, and the
// repository root has plenty that must not go into an APK (.venv, data/,
// tests/). So the pieces the server needs are copied into build/ first and
// that copy is what gets packaged. Sync (not Copy) so removed files disappear.

val stagePython by tasks.registering(Sync::class) {
    description = "Copy the copilot's Python code into the build for Chaquopy."
    from(repoRoot) {
        include("*.py")
        include("copilot/**", "review/**", "storage/**", "stt/**")
        exclude("**/__pycache__/**", "**/*.pyc")
    }
    from(project.file("src/python"))  // android_entry.py
    into(stagedPython)
}

val stageWeb by tasks.registering(Sync::class) {
    description = "Copy templates/ and static/ into the build as app assets."
    from(repoRoot.resolve("templates")) { into("web/templates") }
    from(repoRoot.resolve("static")) { into("web/static") }
    into(stagedAssets)
}

// The Android plugin and Chaquopy create their tasks late, so hook them lazily.
// Gradle 8 fails a build that reads another task's output without declaring
// the dependency, hence the explicit ones on Chaquopy's Python tasks and on
// the asset merge as well as on preBuild.
tasks.configureEach {
    if (name == "preBuild") dependsOn(stagePython, stageWeb)
    if (name.contains("Python") && name != "stagePython") dependsOn(stagePython)
    if (name.startsWith("merge") && name.endsWith("Assets")) dependsOn(stageWeb)
}

// -- android -------------------------------------------------------------

android {
    namespace = "hk.org.sage.meetingcopilot"
    compileSdk = 35

    defaultConfig {
        applicationId = "hk.org.sage.meetingcopilot"
        minSdk = 29          // Android 10 (2019): MediaStore downloads without storage permissions
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        ndk {
            // Chaquopy needs this set. Real phones are all arm64; add "x86_64"
            // only if you want to run the app in the Android Studio emulator
            // (it makes the APK roughly twice as large).
            abiFilters += listOf("arm64-v8a")
        }
    }

    sourceSets {
        getByName("main") {
            assets.srcDir(stagedAssets.get().asFile)
        }
    }

    buildTypes {
        release {
            // No shrinking: Chaquopy's Python is loaded by name, and the app is
            // small enough that there is nothing worth the risk of stripping.
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        buildConfig = true
    }
}

// -- python --------------------------------------------------------------

chaquopy {
    defaultConfig {
        // The build machine must have the same Python major.minor installed
        // (Chaquopy runs pip with it). On Windows it is found as `py -3.12`.
        version = "3.12"

        pip {
            // The exact same dependency list as the PC install. All pure Python,
            // so pip finds wheels for Android without any native compilation.
            install("-r", repoRoot.resolve("requirements.txt").absolutePath)
        }
    }

    sourceSets {
        getByName("main") {
            srcDir(stagedPython.get().asFile.absolutePath)
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.webkit:webkit:1.12.1")
}
