package com.poseadvisor.app.ui

import android.content.Context
import android.graphics.Canvas
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
 *  - 引导姿势：实心人形剪影（参数化解剖模板），等比缩放居中，不变形
 *  - 用户姿势：按 FILL_CENTER 规则与预览画面对齐的实心人体，
 *    达标肢体青色、偏差肢体橙色
 */
class SkeletonOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    private var guideNormalized: FloatArray? = null   // 归一化（躯干单位）引导姿势
    private var guidePixels: FloatArray? = null       // 已适配当前视图尺寸的像素坐标
    private var userPose: FloatArray? = null          // MediaPipe 归一化坐标（图像空间）
    private var segmentOk: BooleanArray? = null
    private var mirror: Boolean = false
    private var imageW = 0                            // 分析图像尺寸（旋转后）
    private var imageH = 0

    // 引导剪影：仅填充（描边会让手臂与躯干交界处的自交线显露出来）
    private val guidePalette = BodyOutlineRenderer.Palette(
        fillColor = 0x705B8CFF,
        strokeColor = 0x00000000,
    )

    fun setGuidePose(pose: FloatArray?) {
        guideNormalized = pose?.let { PoseComparator.normalize(it) }
        guidePixels = null
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

    /** 引导姿势：归一化 → 等比缩放置入屏幕中央安全区（高约 62%，宽不超 85%） */
    private fun fitGuideToView(w: Float, h: Float): FloatArray? {
        val n = guideNormalized ?: return null
        var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
        var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
        for (i in 0 until PoseModel.NUM_JOINTS) {
            minX = minOf(minX, n.jx(i)); maxX = maxOf(maxX, n.jx(i))
            minY = minOf(minY, n.jy(i)); maxY = maxOf(maxY, n.jy(i))
        }
        // 头顶与脚尖在关节之外，留出余量
        minY -= 0.5f; maxY += 0.25f
        val bodyW = (maxX - minX).coerceAtLeast(0.5f)
        val bodyH = (maxY - minY).coerceAtLeast(1f)
        val scale = minOf(0.62f * h / bodyH, 0.85f * w / bodyW)
        val cx = (minX + maxX) / 2; val cy = (minY + maxY) / 2

        val px = FloatArray(PoseModel.POSE_DIM)
        for (i in 0 until PoseModel.NUM_JOINTS) {
            px[i * 3] = (n.jx(i) - cx) * scale + w / 2f
            px[i * 3 + 1] = (n.jy(i) - cy) * scale + h * 0.46f
        }
        return px
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

        // 引导剪影
        if (guidePixels == null) guidePixels = fitGuideToView(w, h)
        guidePixels?.let { px ->
            BodyOutlineRenderer.draw(canvas, px, torsoPx(px), guidePalette)
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
