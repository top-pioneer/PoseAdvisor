package com.poseadvisor.app.camera

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

/**
 * 零快门延迟拍照的核心：最新预览帧的 NV21 字节环形缓冲。
 * 帧到达时仅做一次内存拷贝（毫秒级），按下快门时无需再向相机请求新帧。
 */
class FrameBuffer {

    data class Frame(
        val nv21: ByteArray,
        val width: Int,
        val height: Int,
        val rotationDegrees: Int,
        val timestampMs: Long,
    )

    @Volatile
    var latest: Frame? = null
        private set

    /** 从 ImageProxy 拷贝 YUV_420_888 数据为 NV21 并发布为最新帧 */
    fun accept(image: ImageProxy) {
        val w = image.width
        val h = image.height
        val nv21 = ByteArray(w * h * 3 / 2)
        yuv420ToNv21(image, nv21)
        latest = Frame(nv21, w, h, image.imageInfo.rotationDegrees, System.currentTimeMillis())
    }

    /** 将最新帧解码为处理过旋转的 Bitmap（可选缩放） */
    fun decodeLatest(maxDim: Int = 0): Bitmap? {
        val f = latest ?: return null
        val yuv = YuvImage(f.nv21, ImageFormat.NV21, f.width, f.height, null)
        val out = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, f.width, f.height), 92, out)
        var bmp = BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size()) ?: return null
        if (f.rotationDegrees != 0) {
            val m = Matrix().apply { postRotate(f.rotationDegrees.toFloat()) }
            bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
        }
        if (maxDim > 0 && maxOf(bmp.width, bmp.height) > maxDim) {
            val scale = maxDim.toFloat() / maxOf(bmp.width, bmp.height)
            val m = Matrix().apply { postScale(scale, scale) }
            bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
        }
        return bmp
    }

    companion object {
        /** 兼容任意 rowStride / pixelStride 的 YUV_420_888 → NV21 转换 */
        fun yuv420ToNv21(image: ImageProxy, out: ByteArray) {
            val w = image.width
            val h = image.height
            val yPlane = image.planes[0]
            val uPlane = image.planes[1]
            val vPlane = image.planes[2]

            // Y 平面
            var offset = 0
            val yBuffer = yPlane.buffer
            val yRowStride = yPlane.rowStride
            if (yRowStride == w) {
                yBuffer.get(out, 0, w * h)
                offset = w * h
            } else {
                val row = ByteArray(w)
                for (r in 0 until h) {
                    yBuffer.position(r * yRowStride)
                    yBuffer.get(row, 0, w)
                    System.arraycopy(row, 0, out, offset, w)
                    offset += w
                }
            }

            // VU 平面（NV21 为 V 在前）
            val chromaH = h / 2
            val chromaW = w / 2
            val vBuffer = vPlane.buffer
            val uBuffer = uPlane.buffer
            val vRowStride = vPlane.rowStride
            val uRowStride = uPlane.rowStride
            val vPixelStride = vPlane.pixelStride
            val uPixelStride = uPlane.pixelStride

            for (r in 0 until chromaH) {
                for (c in 0 until chromaW) {
                    out[offset++] = vBuffer.get(r * vRowStride + c * vPixelStride)
                    out[offset++] = uBuffer.get(r * uRowStride + c * uPixelStride)
                }
            }
        }
    }
}
