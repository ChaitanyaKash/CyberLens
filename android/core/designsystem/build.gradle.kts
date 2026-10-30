plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}
android {
    namespace = "io.github.chaitanyakash.cyberlens.core.designsystem"
    compileSdk = 36
    defaultConfig { minSdk = 29 }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true }
}
dependencies {
    implementation("androidx.activity:activity-compose:1.11.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    implementation(platform("androidx.compose:compose-bom:2025.09.01"))
    api("androidx.compose.material3:material3")
    api("androidx.compose.foundation:foundation")
    api("androidx.compose.ui:ui-tooling-preview")
    api("dev.chrisbanes.haze:haze:1.6.10")
}
