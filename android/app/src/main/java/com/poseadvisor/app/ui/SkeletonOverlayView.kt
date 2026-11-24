package com.poseadvisor.app.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.poseadvisor.app.R
import com.poseadvisor.app.pose.PoseComparator
import com.poseadvisor.app.pose.PoseModel
import com.poseadvisor.app.pose.jx
import com.poseadvisor.app.pose.jy
import kotlin.math.max
import kotlin.math.sqrt

/**
 * 姿势叠加渲染层：
 *  - 引导姿势：真人剪影经 MLS 网格变形（MeshPoseRenderer），等比居中
 *  - 用户姿势：按 FILL_CENTER 规则与预览画面对齐的实心人体，
 *    达标肢体青色、偏差肢体橙色
 */
class SkeletonOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    private var guideRaw: FloatArray? = null          // 引导姿势（库原始坐标）
    private var userPose: FloatArray? = null          // MediaPipe 归一化坐标（图像空间）
    private var segmentOk: BooleanArray? = null
    private var mirror: Boolean = false
    private var imageW = 0                            // 分析图像尺寸（旋转后）
    private var imageH = 0

    init {
        MeshPoseRenderer.init(context)
    }

    fun setGuidePose(pose: FloatArray?) {
        guideRaw = pose
        invalidate()
    }

    fun setMirror(m: Boolean) {
        mirror = m
    }

    /** 设置姿态分析所用图像的尺寸（用于用户骨架与预览画面的 FILL_CENTER 对齐） */
    fun setImageSize(w: Int, h: Int) {
        if (w != imageW || h != imageH) {
            imageW = w; imageH = h
        }
    }

    fun updateUserPose(pose: FloatArray?, compare: PoseComparator.CompareResult?) {
        userPose = pose
        segmentOk = compare?.segmentOk
        invalidate()
    }

    /** 用户姿势：图像归一化坐标 → 按 FILL_CENTER 裁剪规则映射到视图像素 */
    private fun userToPixels(pose: FloatArray, w: Float, h: Float): FloatArray {
        val iw = if (imageW > 0) imageW.toFloat() else w
        val ih = if (imageH > 0) imageH.toFloat() else h
        val scale = max(w / iw, h / ih)
        val dx = (w - iw * scale) / 2f
        val dy = (h - ih * scale) / 2f
        val px = FloatArray(PoseModel.POSE_DIM)
        for (i in 0 until PoseModel.NUM_JOINTS) {
            val x = if (mirror) 1f - pose.jx(i) else pose.jx(i)
            px[i * 3] = x * iw * scale + dx
            px[i * 3 + 1] = pose.jy(i) * ih * scale + dy
        }
        return px
    }

    private fun torsoPx(p: FloatArray): Float {
        val dx = p.jx(PoseModel.NECK) - p.jx(PoseModel.PELVIS)
        val dy = p.jy(PoseModel.NECK) - p.jy(PoseModel.PELVIS)
        return sqrt(dx * dx + dy * dy).coerceAtLeast(1f)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0 || h <= 0) return

        // 引导剪影（真人轮廓，网格变形）
        guideRaw?.let { g ->
            val aspect = MeshPoseRenderer.bitmapAspect()
            val rh = h * 0.62f
            val rw = if (aspect > 0) rh * aspect else rh * 0.45f
            val rect = RectF(w / 2f - rw / 2f, h * 0.46f - rh / 2f, w / 2f + rw / 2f, h * 0.46f + rh / 2f)
            MeshPoseRenderer.draw(canvas, g, rect, 0xFF5B8CFF.toInt(), 0xB0)
        }

        // 用户实时人体（分段着色）
        val u = userPose ?: return
        val px = userToPixels(u, w, h)
        val ok = segmentOk
        val matchColor = ContextCompat.getColor(context, R.color.pose_match)
        val mismatchColor = ContextCompat.getColor(context, R.color.pose_mismatch)
        // 比对段顺序：躯干/左上臂/左前臂/右上臂/右前臂/左腿/右腿
        val limbColors = IntArray(6) { i ->
            val okSeg = ok == null || ok[i + 1]
            if (okSeg) matchColor else mismatchColor
        }
        val torsoOk = ok == null || ok[0]
        BodyOutlineRenderer.drawUser(
            canvas, px, torsoPx(px),
            torsoColor = (if (torsoOk) matchColor else mismatchColor) and 0x00FFFFFF or (0x88 shl 24),
            headColor = (if (torsoOk) matchColor else mismatchColor) and 0x00FFFFFF or (0x88 shl 24),
            limbColors = limbColors,
        )
    }
}
