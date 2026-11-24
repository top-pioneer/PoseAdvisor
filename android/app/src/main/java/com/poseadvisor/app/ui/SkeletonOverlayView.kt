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

/**
 * 姿势叠加渲染层（丰满人形轮廓版）：
 *  - 引导姿势：半透明品牌蓝实心人形剪影，所见即目标
 *  - 用户姿势：实心轮廓，达标肢体青色、偏差肢体橙色，逐段即时反馈
 */
class SkeletonOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    private var guidePose: FloatArray? = null      // 已转为屏幕归一化坐标
    private var userPose: FloatArray? = null       // MediaPipe 原始归一化坐标
    private var segmentOk: BooleanArray? = null    // 7 段：躯干/双臂/双腿
    private var mirror: Boolean = false

    private val guidePalette = BodyOutlineRenderer.Palette(
        fillColor = 0x555B8CFF,
        strokeColor = 0xCC5B8CFF.toInt(),
    )

    fun setGuidePose(pose: FloatArray?) {
        guidePose = pose?.let { PoseComparator.guideForDisplay(it) }
        invalidate()
    }

    fun setMirror(m: Boolean) {
        mirror = m
    }

    fun updateUserPose(pose: FloatArray?, compare: PoseComparator.CompareResult?) {
        userPose = pose
        segmentOk = compare?.segmentOk
        invalidate()
    }

    /** 屏幕归一化坐标 → 像素坐标 */
    private fun toPixels(pose: FloatArray, w: Float, h: Float, mirrorX: Boolean): FloatArray {
        val px = FloatArray(PoseModel.POSE_DIM)
        for (i in 0 until PoseModel.NUM_JOINTS) {
            val x = pose.jx(i)
            px[i * 3] = (if (mirrorX) 1f - x else x) * w
            px[i * 3 + 1] = pose.jy(i) * h
            px[i * 3 + 2] = pose[i * 3 + 2]
        }
        return px
    }

    private fun torsoPx(p: FloatArray): Float {
        val dx = p.jx(PoseModel.NECK) - p.jx(PoseModel.PELVIS)
        val dy = p.jy(PoseModel.NECK) - p.jy(PoseModel.PELVIS)
        return kotlin.math.sqrt(dx * dx + dy * dy).coerceAtLeast(1f)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0 || h <= 0) return

        // 引导剪影
        guidePose?.let { g ->
            val px = toPixels(g, w, h, mirrorX = false)
            BodyOutlineRenderer.draw(canvas, px, torsoPx(px), guidePalette)
        }

        // 用户实时轮廓（按肢体段着色）
        val u = userPose ?: return
        val px = toPixels(u, w, h, mirrorX = mirror)
        val ok = segmentOk
        val matchColor = ContextCompat.getColor(context, R.color.pose_match)
        val mismatchColor = ContextCompat.getColor(context, R.color.pose_mismatch)
        // 渲染器四肢顺序：左上臂/左前臂/右上臂/右前臂/左腿/右腿
        // 比对段顺序：躯干/左上臂/左前臂/右上臂/右前臂/左腿/右腿 → 去掉躯干
        val boneColors = IntArray(6) { i ->
            val okSeg = ok == null || ok[i + 1]
            if (okSeg) matchColor else mismatchColor
        }
        val torsoOk = ok == null || ok[0]
        BodyOutlineRenderer.draw(
            canvas, px, torsoPx(px),
            BodyOutlineRenderer.Palette(
                fillColor = (if (torsoOk) matchColor else mismatchColor) and 0x00FFFFFF or (0xAA shl 24),
                strokeColor = 0x00000000,
            ),
            perBoneColors = boneColors,
        )
    }
}
