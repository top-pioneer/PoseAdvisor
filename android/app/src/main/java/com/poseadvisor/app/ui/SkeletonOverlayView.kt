package com.poseadvisor.app.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.poseadvisor.app.R
import com.poseadvisor.app.pose.PoseComparator
import com.poseadvisor.app.pose.PoseModel
import com.poseadvisor.app.pose.jx
import com.poseadvisor.app.pose.jy

/**
 * 姿势叠加渲染层：
 *  - 半透明品牌蓝"幽灵骨架"：用户要达成的推荐姿势
 *  - 用户实时骨架：达标肢体青色、偏差肢体橙色，逐段即时反馈
 */
class SkeletonOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    private var guidePose: FloatArray? = null      // 已转为屏幕归一化坐标
    private var userPose: FloatArray? = null       // MediaPipe 原始归一化坐标
    private var segmentOk: BooleanArray? = null
    private var mirror: Boolean = false

    private val guideLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.pose_guide)
        strokeWidth = 9f
        strokeCap = Paint.Cap.ROUND
        style = Paint.Style.STROKE
    }
    private val guideJointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.brand_primary)
        style = Paint.Style.FILL
        alpha = 160
    }
    private val matchPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.pose_match)
        strokeWidth = 8f
        strokeCap = Paint.Cap.ROUND
        style = Paint.Style.STROKE
    }
    private val mismatchPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.pose_mismatch)
        strokeWidth = 8f
        strokeCap = Paint.Cap.ROUND
        style = Paint.Style.STROKE
    }
    private val userJointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.white)
        style = Paint.Style.FILL
    }

    /** 设置引导姿势（库中原始坐标，内部转为展示坐标） */
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

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()

        guidePose?.let { g ->
            for ((a, b) in PoseModel.RENDER_EDGES) {
                canvas.drawLine(g.jx(a) * w, g.jy(a) * h, g.jx(b) * w, g.jy(b) * h, guideLinePaint)
            }
            for (i in 0 until PoseModel.NUM_JOINTS) {
                if (g.jx(i) == 0f && g.jy(i) == 0f) continue
                canvas.drawCircle(g.jx(i) * w, g.jy(i) * h, 7f, guideJointPaint)
            }
        }

        val u = userPose ?: return
        fun ux(i: Int): Float {
            val x = u.jx(i)
            return (if (mirror) 1f - x else x) * w
        }
        fun uy(i: Int) = u.jy(i) * h

        val segs = PoseModel.COMPARE_SEGMENTS
        val ok = segmentOk
        // 先画参与比对的肢体段（带颜色反馈）
        for (i in segs.indices) {
            val s = segs[i]
            val paint = if (ok == null || ok[i]) matchPaint else mismatchPaint
            canvas.drawLine(ux(s.a), uy(s.a), ux(s.b), uy(s.b), paint)
        }
        // 再画其余连接（细白）
        for ((a, b) in PoseModel.RENDER_EDGES) {
            if (segs.any { it.a == a && it.b == b }) continue
            canvas.drawLine(ux(a), uy(a), ux(b), uy(b), matchPaint.apply { alpha = 90 })
        }
        matchPaint.alpha = 255
        for (i in 0 until PoseModel.NUM_JOINTS) {
            if (u.jx(i) == 0f && u.jy(i) == 0f) continue
            canvas.drawCircle(ux(i), uy(i), 5f, userJointPaint)
        }
    }
}
