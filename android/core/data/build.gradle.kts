plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.kapt")
    id("com.google.dagger.hilt.android")
}
android {
    namespace = "io.github.chaitanyakash.cyberlens.core.data"
    compileSdk = 36
    defaultConfig { minSdk = 29 }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
}
kapt { arguments { arg("room.schemaLocation", file("schemas").path) } }
dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    api("androidx.room:room-runtime:2.8.1")
    implementation("androidx.room:room-ktx:2.8.1")
    kapt("androidx.room:room-compiler:2.8.1")
    implementation("androidx.datastore:datastore-preferences:1.1.7")
    implementation("net.zetetic:sqlcipher-android:4.10.0")
    implementation("androidx.sqlite:sqlite:2.6.1")
    implementation("com.google.dagger:hilt-android:2.57.1")
    kapt("com.google.dagger:hilt-compiler:2.57.1")
}
