plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("org.jetbrains.kotlin.kapt")
    id("com.google.dagger.hilt.android")
    id("io.github.takahirom.roborazzi")
}
android {
    namespace = "io.github.chaitanyakash.cyberlens"
    compileSdk = 36
    defaultConfig {
        applicationId = "io.github.chaitanyakash.cyberlens"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    buildFeatures { compose = true; buildConfig = true }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
    testOptions {
        unitTests.isIncludeAndroidResources = true
        unitTests.all {
            it.maxParallelForks = 1
            it.maxHeapSize = "1536m"
            it.jvmArgs("-XX:ActiveProcessorCount=2", "-XX:TieredStopAtLevel=1", "-XX:ReservedCodeCacheSize=128m")
            it.systemProperty("robolectric.graphicsMode", "NATIVE")
            it.systemProperty("cyberlens.screenshot.smoke", providers.gradleProperty("screenshotSmoke").getOrElse("false"))
            it.systemProperty("cyberlens.screenshot.window", providers.gradleProperty("screenshotWindow").getOrElse(""))
        }
    }
    buildTypes { release { isMinifyEnabled = false } }
}
kapt { correctErrorTypes = true }
dependencies {
    implementation(project(":core:designsystem"))
    implementation(project(":core:data"))
    implementation(project(":core:common"))
    implementation(platform("androidx.compose:compose-bom:2025.09.01"))
    implementation("androidx.activity:activity-compose:1.11.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.9.4")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.9.4")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material3:material3-adaptive-navigation-suite")
    implementation("androidx.compose.material:material-icons-core")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("com.google.dagger:hilt-android:2.57.1")
    kapt("com.google.dagger:hilt-compiler:2.57.1")
    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
    testImplementation(project(":core:testing"))
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.robolectric:robolectric:4.16")
    testImplementation("androidx.compose.ui:ui-test-junit4")
    testImplementation("io.github.takahirom.roborazzi:roborazzi:1.76.0")
    testImplementation("io.github.takahirom.roborazzi:roborazzi-compose:1.76.0")
    testImplementation("io.github.takahirom.roborazzi:roborazzi-junit-rule:1.76.0")
}
