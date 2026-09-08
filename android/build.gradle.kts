// Versions chosen to sit inside every range that has to overlap:
//   Chaquopy 17.0 supports Android Gradle Plugin 7.3 - 9.3 and Python 3.10 - 3.14.
//   AGP 8.7 needs Gradle 8.9+ (see gradle/wrapper/gradle-wrapper.properties).
// If Android Studio offers to "upgrade the Android Gradle Plugin", decline:
// anything above 9.3 is outside what Chaquopy 17 accepts.
plugins {
    id("com.android.application") version "8.7.3" apply false
    id("org.jetbrains.kotlin.android") version "2.0.21" apply false
    id("com.chaquo.python") version "17.0.0" apply false
}
