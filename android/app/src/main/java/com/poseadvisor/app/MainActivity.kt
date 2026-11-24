package com.poseadvisor.app

import android.Manifest
import android.animation.ObjectAnimator
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
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
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import com.poseadvisor.app.camera.CameraController
import com.poseadvisor.app.camera.FrameBuffer
import com.poseadvisor.app.pose.PoseComparator
import com.poseadvisor.app.pose.PoseLandmarkerHelper
import com.poseadvisor.app.pose.PoseLibrary
import com.poseadvisor.app.scene.SceneClassifier
import com.poseadvisor.app.ui.PoseStripAdapter
import com.poseadvisor.app.ui.SkeletonOverlayView
import java.io.File
import java.io.FileOutputStream

/**
 * PoseAdvisor 主界面：实时取景 + 姿势推荐 + 匹配引导 + 零快门延迟拍照。
 */
class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: SkeletonOverlayView
    private lateinit var scoreBadge: TextView
    private lateinit var hintText: TextView
    private lateinit var sceneChip: TextView
    private lateinit var flashView: View
    private lateinit var lastPhotoThumb: ImageView
    private lateinit var categoryChips: ChipGroup
    private lateinit var poseStrip: RecyclerView

    private val frameBuffer = FrameBuffer()
    private lateinit var cameraController: CameraController
    private var poseHelper: PoseLandmarkerHelper? = null
    private lateinit var library: PoseLibrary
    private lateinit var stripAdapter: PoseStripAdapter

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
        WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS.let { window.addFlags(it) }
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        scoreBadge = findViewById(R.id.scoreBadge)
        hintText = findViewById(R.id.hintText)
        sceneChip = findViewById(R.id.sceneChip)
        flashView = findViewById(R.id.flashView)
        lastPhotoThumb = findViewById(R.id.lastPhotoThumb)
        categoryChips = findViewById(R.id.categoryChips)
        poseStrip = findViewById(R.id.poseStrip)

        library = PoseLibrary.load(this)
        setupCategoryChips()
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
        lastPhotoThumb.setOnClickListener {
            lastPhotoUri?.let { uri ->
                startActivity(Intent(Intent.ACTION_VIEW).setDataAndType(uri, "image/jpeg"))
            }
        }

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
        // 启动后做一次自动场景识别，给出初始推荐类别
        previewView.postDelayed({ detectScene() }, 1200)
    }

    // ---------- 姿势推荐 UI ----------

    private fun setupCategoryChips() {
        for (cat in library.categories) {
            val chip = Chip(this).apply {
                text = cat.label
                tag = cat.id
                isCheckable = true
                setTextColor(ContextCompat.getColor(context, R.color.text_primary))
                chipBackgroundColor = ContextCompat.getColorStateList(context, R.color.surface_chip)
                setOnClickListener { selectCategory(cat.id) }
            }
            categoryChips.addView(chip)
        }
        (categoryChips.getChildAt(0) as? Chip)?.isChecked = true
    }

    private fun setupPoseStrip() {
        stripAdapter = PoseStripAdapter { entry ->
            overlayView.setGuidePose(entry.joints)
        }
        poseStrip.layoutManager = LinearLayoutManager(this, LinearLayoutManager.HORIZONTAL, false)
        poseStrip.adapter = stripAdapter
        selectCategory(library.categories.first().id)
    }

    private fun selectCategory(categoryId: String) {
        for (i in 0 until categoryChips.childCount) {
            (categoryChips.getChildAt(i) as? Chip)?.isChecked =
                categoryChips.getChildAt(i).tag == categoryId
        }
        val list = library.posesOf(categoryId)
        stripAdapter.submit(list)
        overlayView.setGuidePose(stripAdapter.current()?.joints)
    }

    // ---------- 实时姿态推理与匹配反馈 ----------

    private fun maybeRunPoseInference() {
        frameCounter++
        if (frameCounter % 3 != 0) return          // 约 10fps 推理频率
        if (poseBusy) return
        poseBusy = true
        Thread {
            try {
                val bmp = frameBuffer.decodeLatest(maxDim = 480) ?: return@Thread
                poseHelper?.detectAsync(bmp, System.currentTimeMillis())
            } finally {
                poseBusy = false
            }
        }.start()
    }

    private fun onUserPose(joints: FloatArray) {
        val guide = stripAdapter.current()?.joints
        if (guide == null) {
            overlayView.updateUserPose(joints, null)
            return
        }
        val result = PoseComparator.compare(joints, guide)
        overlayView.updateUserPose(joints, result)

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

    // ---------- 场景识别 ----------

    private fun detectScene() {
        val frame = frameBuffer.latest ?: return
        val hint = SceneClassifier.classify(frame)
        sceneChip.text = hint.label
        sceneChip.visibility = View.VISIBLE
        selectCategory(hint.suggestedCategory)
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

    private fun saveToGallery(bmp: Bitmap): Uri? {
        val name = "PoseAdvisor_${System.currentTimeMillis()}.jpg"
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, name)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/PoseAdvisor")
            }
            val uri = contentResolver.insert(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values
            )
            uri?.also {
                contentResolver.openOutputStream(it)?.use { out ->
                    bmp.compress(Bitmap.CompressFormat.JPEG, 95, out)
                }
            }
            uri
        } else {
            @Suppress("DEPRECATION")
            val dir = File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
                "PoseAdvisor"
            )
            dir.mkdirs()
            val file = File(dir, name)
            FileOutputStream(file).use { out ->
                bmp.compress(Bitmap.CompressFormat.JPEG, 95, out)
            }
            MediaScannerConnection.scanFile(this, arrayOf(file.absolutePath), null, null)
            Uri.fromFile(file)
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
