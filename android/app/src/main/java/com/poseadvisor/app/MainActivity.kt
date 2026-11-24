package com.poseadvisor.app

import android.Manifest
import android.animation.ObjectAnimator
import android.content.ActivityNotFoundException
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.view.WindowManager
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.poseadvisor.app.camera.CameraController
import com.poseadvisor.app.camera.FrameBuffer
import com.poseadvisor.app.pose.PoseComparator
import com.poseadvisor.app.pose.PoseEntry
import com.poseadvisor.app.pose.PoseLandmarkerHelper
import com.poseadvisor.app.pose.PoseLibrary
import com.poseadvisor.app.pose.PoseRecommender
import com.poseadvisor.app.pose.PoseSynthesizer
import com.poseadvisor.app.scene.SceneClassifier
import com.poseadvisor.app.ui.BodyOutlineRenderer
import com.poseadvisor.app.ui.PoseStripAdapter
import com.poseadvisor.app.ui.SkeletonOverlayView

/**
 * PoseAdvisor 主界面：实时取景 + 智能姿势推荐 + 匹配引导 + 零快门延迟拍照。
 *
 * 智能推荐状态机：
 * 场景识别 → 姿势亲和度排序 → 构建推荐队列（库姿势 + AI 合成新姿势）→
 * 自动轮换引导（用户长时间未达标则换下一个；用户点卡则锁定 60s）。
 */
class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: SkeletonOverlayView
    private lateinit var scoreBadge: TextView
    private lateinit var hintText: TextView
    private lateinit var sceneChip: TextView
    private lateinit var flashView: View
    private lateinit var lastPhotoThumb: ImageView
    private lateinit var poseStrip: RecyclerView

    private val frameBuffer = FrameBuffer()
    private lateinit var cameraController: CameraController
    private var poseHelper: PoseLandmarkerHelper? = null
    private lateinit var library: PoseLibrary
    private lateinit var stripAdapter: PoseStripAdapter

    // ---- 智能推荐状态 ----
    /** 推荐队列项（库姿势或 AI 合成姿势 + 卡片缩略图） */
    private data class QueueItem(val entry: PoseEntry, val thumb: Bitmap?)

    private val guideQueue = mutableListOf<QueueItem>()
    private var queueIndex = -1
    private var currentSinceMs = 0L
    private var bestScoreCurrent = 0
    private var lockUntilMs = 0L               // 用户手动选卡后的锁定截止时间
    private var synthCounter = 0

    private var frameCounter = 0
    private var lastPhotoUri: Uri? = null
    @Volatile private var poseBusy = false

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else {
            Toast.makeText(this, R.string.camera_permission_rationale, Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        scoreBadge = findViewById(R.id.scoreBadge)
        hintText = findViewById(R.id.hintText)
        sceneChip = findViewById(R.id.sceneChip)
        flashView = findViewById(R.id.flashView)
        lastPhotoThumb = findViewById(R.id.lastPhotoThumb)
        poseStrip = findViewById(R.id.poseStrip)

        library = PoseLibrary.load(this)
        setupPoseStrip()

        cameraController = CameraController(this, this, previewView) { image ->
            frameBuffer.accept(image)
            maybeRunPoseInference()
        }

        poseHelper = PoseLandmarkerHelper(this) { joints ->
            runOnUiThread { onUserPose(joints) }
        }

        findViewById<ImageButton>(R.id.btnSwitchCamera).setOnClickListener {
            cameraController.switchCamera()
            overlayView.setMirror(cameraController.isFront())
        }
        findViewById<ImageView>(R.id.btnShutter).setOnClickListener { captureZeroLag() }
        findViewById<ImageButton>(R.id.btnDetectScene).setOnClickListener { detectScene() }
        lastPhotoThumb.setOnClickListener { openLastPhoto() }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        cameraController.start()
        overlayView.setMirror(cameraController.isFront())
        // 启动后自动场景识别 → 自动生成推荐队列
        previewView.postDelayed({ detectScene() }, 1200)
    }

    // ---------- 智能推荐 ----------

    private fun setupPoseStrip() {
        stripAdapter = PoseStripAdapter { entry ->
            // 用户手动选卡：锁定 60 秒，暂停自动轮换
            lockUntilMs = System.currentTimeMillis() + 60_000
            val idx = guideQueue.indexOfFirst { it.entry.id == entry.id }
            if (idx >= 0) setCurrentGuide(idx)
        }
        poseStrip.layoutManager = LinearLayoutManager(this, LinearLayoutManager.HORIZONTAL, false)
        poseStrip.adapter = stripAdapter
    }

    /** 场景识别 → 亲和度排序 → 构建"库姿势 + AI 合成"混合推荐队列 */
    private fun detectScene() {
        val frame = frameBuffer.latest ?: return
        val hint = SceneClassifier.classify(frame)
        sceneChip.text = hint.label
        sceneChip.visibility = View.VISIBLE
        buildGuideQueue(hint.profile)
    }

    private fun buildGuideQueue(profile: SceneClassifier.SceneProfile) {
        val ranked = PoseRecommender.rank(library, profile)
        guideQueue.clear()
        synthCounter = 0

        val seeds = ranked.take(4).map { it.entry }
        // 前 3 个最契合的库姿势先行；随后穿插 AI 合成新姿势
        seeds.take(3).forEach { guideQueue.add(QueueItem(it, renderCardThumb(it.joints))) }
        if (seeds.size >= 2) addSynthesized(seeds[0], seeds[1])
        if (seeds.size >= 4) {
            guideQueue.add(QueueItem(seeds[3], renderCardThumb(seeds[3].joints)))
            addSynthesized(seeds[1], seeds[3])
        }

        stripAdapter.submit(guideQueue.map { PoseStripAdapter.Item(it.entry, it.thumb) })
        setCurrentGuide(0)
        Toast.makeText(this, "已按场景智能推荐姿势", Toast.LENGTH_SHORT).show()
    }

    /** 用解剖渲染器统一生成姿势卡片缩略图（库姿势与 AI 姿势风格一致） */
    private fun renderCardThumb(joints: FloatArray): Bitmap {
        return BodyOutlineRenderer.renderThumbnail(
            joints, 192,
            BodyOutlineRenderer.Palette(0xFF5B8CFF.toInt(), 0x00000000),
            0xFF171A21.toInt(),
        )
    }

    /** 用两个种子姿势合成新姿势，渲染缩略图后加入队列 */
    private fun addSynthesized(a: PoseEntry, b: PoseEntry) {
        val joints = PoseSynthesizer.synthesize(a.joints, b.joints)
        synthCounter++
        val entry = PoseEntry(
            id = "ai_synth_$synthCounter",
            category = "ai",
            label = "AI 新创·$synthCounter",
            thumbnail = "",
            joints = joints,
        )
        val thumb = renderCardThumb(joints)
        guideQueue.add(QueueItem(entry, thumb))
    }

    private fun setCurrentGuide(index: Int) {
        if (index !in guideQueue.indices) return
        queueIndex = index
        currentSinceMs = System.currentTimeMillis()
        bestScoreCurrent = 0
        val item = guideQueue[index]
        overlayView.setGuidePose(item.entry.joints)
        stripAdapter.selectById(item.entry.id)
        // 卡条顺序与队列顺序不完全一致，滚动到相近位置即可
        poseStrip.smoothScrollToPosition(index.coerceIn(0, guideQueue.size - 1))
    }

    /** 自动轮换：用户在该姿势下长时间未达标 → 换下一个候选 */
    private fun maybeAdvanceGuide(score: Int) {
        val now = System.currentTimeMillis()
        if (now < lockUntilMs) return
        if (score > bestScoreCurrent) bestScoreCurrent = score
        if (score >= 85) return // 已达标，保持当前姿势
        val dwellMs = now - currentSinceMs
        // 12 秒无进展，或 8 秒仍低于 40 分 → 换下一个
        if (dwellMs > 12_000 || (dwellMs > 8_000 && bestScoreCurrent < 40)) {
            val next = queueIndex + 1
            if (next < guideQueue.size) {
                setCurrentGuide(next)
            } else {
                // 队列用尽：以当前最优种子再合成一批新姿势
                val seeds = guideQueue.take(3).map { it.entry }
                if (seeds.size >= 2) {
                    addSynthesized(seeds[0], seeds[1])
                    addSynthesized(seeds[1], seeds.last())
                    stripAdapter.submit(guideQueue.map { PoseStripAdapter.Item(it.entry, it.thumb) })
                    setCurrentGuide(next)
                } else {
                    setCurrentGuide(0)
                }
            }
        }
    }

    // ---------- 实时姿态推理与匹配反馈 ----------

    private fun maybeRunPoseInference() {
        frameCounter++
        if (frameCounter % 3 != 0) return
        if (poseBusy) return
        poseBusy = true
        Thread {
            try {
                val bmp = frameBuffer.decodeLatest(maxDim = 480) ?: return@Thread
                // 告知叠加层分析图像尺寸，保证用户骨架与预览画面严格对齐
                overlayView.setImageSize(bmp.width, bmp.height)
                poseHelper?.detectAsync(bmp, System.currentTimeMillis())
            } finally {
                poseBusy = false
            }
        }.start()
    }

    private fun onUserPose(joints: FloatArray) {
        val guide = guideQueue.getOrNull(queueIndex)?.entry?.joints
        if (guide == null) {
            overlayView.updateUserPose(joints, null)
            return
        }
        val result = PoseComparator.compare(joints, guide)
        overlayView.updateUserPose(joints, result)
        maybeAdvanceGuide(result.score)

        scoreBadge.visibility = View.VISIBLE
        scoreBadge.text = getString(R.string.score_label) + " " + result.score
        scoreBadge.setTextColor(
            ContextCompat.getColor(
                this,
                if (result.score >= 80) R.color.pose_match else R.color.pose_mismatch
            )
        )
        if (result.hint != null) {
            hintText.text = result.hint
            hintText.visibility = View.VISIBLE
        } else {
            hintText.text = getString(R.string.hint_perfect)
            hintText.visibility = if (result.score >= 85) View.VISIBLE else View.GONE
        }
    }

    // ---------- 零快门延迟拍照 ----------

    private fun captureZeroLag() {
        val t0 = System.currentTimeMillis()
        flashAnimation()
        Thread {
            val bmp = frameBuffer.decodeLatest(maxDim = 1920)
            if (bmp == null) {
                runOnUiThread { Toast.makeText(this, "取景未就绪", Toast.LENGTH_SHORT).show() }
                return@Thread
            }
            val finalBmp = if (cameraController.isFront()) mirrorBitmap(bmp) else bmp
            val uri = saveToGallery(finalBmp)
            val elapsed = System.currentTimeMillis() - t0
            runOnUiThread { onCaptured(uri, finalBmp, elapsed) }
        }.start()
    }

    private fun mirrorBitmap(bmp: Bitmap): Bitmap {
        val m = android.graphics.Matrix().apply { preScale(-1f, 1f) }
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, false)
    }

    /** 统一走 MediaStore 保存（API 26+ 均适用，无需文件权限） */
    private fun saveToGallery(bmp: Bitmap): Uri? {
        val name = "PoseAdvisor_${System.currentTimeMillis()}.jpg"
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/PoseAdvisor")
            }
        }
        val collection = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        } else {
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI
        }
        val uri = contentResolver.insert(collection, values) ?: return null
        return try {
            contentResolver.openOutputStream(uri)?.use { out ->
                bmp.compress(Bitmap.CompressFormat.JPEG, 95, out)
            } ?: return null
            uri
        } catch (e: Exception) {
            contentResolver.delete(uri, null, null)
            null
        }
    }

    private fun onCaptured(uri: Uri?, bmp: Bitmap, elapsedMs: Long) {
        if (uri == null) {
            Toast.makeText(this, "保存失败", Toast.LENGTH_SHORT).show()
            return
        }
        lastPhotoUri = uri
        lastPhotoThumb.setImageBitmap(bmp)
        lastPhotoThumb.visibility = View.VISIBLE
        Toast.makeText(this, "${getString(R.string.capture_saved)}（${elapsedMs}ms）", Toast.LENGTH_SHORT).show()
    }

    /** 预览最近照片：授予临时读取权限，兼容无图库应用的情况 */
    private fun openLastPhoto() {
        val uri = lastPhotoUri ?: return
        val intent = Intent(Intent.ACTION_VIEW).apply {
            setDataAndType(uri, "image/jpeg")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        try {
            startActivity(intent)
        } catch (e: ActivityNotFoundException) {
            Toast.makeText(this, "未找到可查看图片的应用", Toast.LENGTH_SHORT).show()
        }
    }

    private fun flashAnimation() {
        ObjectAnimator.ofFloat(flashView, View.ALPHA, 0f, 0.85f, 0f).apply {
            duration = 260
            start()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        poseHelper?.close()
        cameraController.shutdown()
    }
}
