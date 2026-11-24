package com.poseadvisor.app.pose

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker

/**
 * MediaPipe PoseLandmarker 封装：GPU 优先、CPU 兜底，LIVE_STREAM 异步推理。
 */
class PoseLandmarkerHelper(
    context: Context,
    private val onResult: (joints75: FloatArray) -> Unit,
) {
    private var landmarker: PoseLandmarker? = null

    init {
        val baseBuilder = BaseOptions.builder()
            .setModelAssetPath("pose_landmarker_lite.task")
        try {
            baseBuilder.setDelegate(Delegate.GPU)
            landmarker = create(context, baseBuilder.build())
        } catch (e: Exception) {
            Log.w(TAG, "GPU 委托不可用，回退 CPU", e)
            try {
                baseBuilder.setDelegate(Delegate.CPU)
                landmarker = create(context, baseBuilder.build())
            } catch (e2: Exception) {
                Log.e(TAG, "PoseLandmarker 初始化失败", e2)
            }
        }
    }

    private fun create(context: Context, base: BaseOptions): PoseLandmarker {
        val options = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumPoses(1)
            .setMinPoseDetectionConfidence(0.5f)
            .setMinPosePresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setResultListener { result, _ ->
                PoseMapper.to75d(result)?.let(onResult)
            }
            .setErrorListener { e -> Log.e(TAG, "推理错误: ${e.message}") }
            .build()
        return PoseLandmarker.createFromOptions(context, options)
    }

    fun detectAsync(bitmap: Bitmap, timestampMs: Long) {
        val lm = landmarker ?: return
        try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            lm.detectAsync(mpImage, timestampMs)
        } catch (e: Exception) {
            Log.e(TAG, "detectAsync 失败", e)
        }
    }

    fun close() {
        landmarker?.close()
        landmarker = null
    }

    companion object {
        private const val TAG = "PoseLandmarkerHelper"
    }
}
