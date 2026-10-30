package io.github.chaitanyakash.cyberlens.core.data

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import androidx.datastore.preferences.core.*
import androidx.datastore.preferences.preferencesDataStore
import androidx.room.*
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import java.security.KeyStore
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.map
import net.zetetic.database.sqlcipher.SupportOpenHelperFactory

@Entity(tableName = "incidents")
data class IncidentEntity(@PrimaryKey val id: String, val createdAt: Long, val status: String)
@Dao
interface IncidentDao {
    @Query("SELECT * FROM incidents ORDER BY createdAt DESC")
    fun observe(): kotlinx.coroutines.flow.Flow<List<IncidentEntity>>
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun save(incident: IncidentEntity)
}
@Database(entities = [IncidentEntity::class], version = 1, exportSchema = true)
abstract class CyberLensDatabase : RoomDatabase() {
    abstract fun incidents(): IncidentDao
}

private val Context.settingsStore by preferencesDataStore("appearance")
@Singleton
class AppearanceSettings @Inject constructor(@ApplicationContext private val context: Context) {
    private val motionKey = booleanPreferencesKey("reduce_motion")
    private val transparencyKey = booleanPreferencesKey("reduce_transparency")
    val motion = context.settingsStore.data.map { it[motionKey] ?: false }
    val transparency = context.settingsStore.data.map { it[transparencyKey] ?: false }
    suspend fun setMotion(value: Boolean) { context.settingsStore.edit { it[motionKey] = value } }
    suspend fun setTransparency(value: Boolean) { context.settingsStore.edit { it[transparencyKey] = value } }
}

@Singleton
class DatabaseKey @Inject constructor(@ApplicationContext private val context: Context) {
    @Synchronized
    fun bytes(): ByteArray {
        val keyStore = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        val alias = "cyberlens.database"
        val key = (keyStore.getKey(alias, null) as? SecretKey) ?: KeyGenerator.getInstance(
            KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore",
        ).apply {
            init(KeyGenParameterSpec.Builder(alias, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE).build())
        }.generateKey()
        val file = java.io.File(context.noBackupFilesDir, "database.key")
        if (file.exists()) {
            val blob = file.readBytes()
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            cipher.init(Cipher.DECRYPT_MODE, key, GCMParameterSpec(128, blob.copyOfRange(0, 12)))
            return cipher.doFinal(blob.copyOfRange(12, blob.size))
        }
        val secret = ByteArray(32).also { SecureRandom().nextBytes(it) }
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, key)
        val temporary = java.io.File(context.noBackupFilesDir, "database.key.tmp")
        temporary.writeBytes(cipher.iv + cipher.doFinal(secret))
        check(temporary.renameTo(file)) { "Unable to save wrapped database key" }
        return secret
    }
}

@Module
@InstallIn(SingletonComponent::class)
object DataModule {
    @Provides @Singleton
    fun database(@ApplicationContext context: Context, key: DatabaseKey): CyberLensDatabase {
        System.loadLibrary("sqlcipher")
        return Room.databaseBuilder(context, CyberLensDatabase::class.java, "cyberlens.db")
            .openHelperFactory(SupportOpenHelperFactory(key.bytes())).build()
    }
}
