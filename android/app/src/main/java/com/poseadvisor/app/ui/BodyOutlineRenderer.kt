package com.poseadvisor.app.ui

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import com.poseadvisor.app.pose.PoseModel
import com.poseadvisor.app.pose.jx
import com.poseadvisor.app.pose.jy

/**
 * 丰满人形轮廓渲染器：
 * 把 25 关节骨架"长肉"成实心剪影——头部圆、躯干圆角梯形、四肢胶囊体，
 * 让引导姿势像真实人形轮廓而非抽象骨架，更易于对照摆姿。
 */
object BodyOutlineRenderer {

    data class Palette(
        val fillColor: Int,
        val strokeColor: Int,
    )

    /**
     * 将姿势绘制为丰满人形轮廓。
     * @param pose 屏幕像素坐标的 75 维向量（调用方负责坐标变换）
     * @param torsoRef 参考躯干长度（像素），决定肢体粗细
     * @param perBoneColors 可选：按骨骼索引（BONES 顺序）给四肢单独着色（用户实时反馈用）
     */
    fun draw(
        canvas: Canvas,
        pose: FloatArray,
        torsoRef: Float,
        palette: Palette,
        perBoneColors: IntArray? = null,
    ) {
        val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = palette.fillColor
            style = Paint.Style.FILL
        }
        val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = palette.strokeColor
            style = Paint.Style.STROKE
            strokeWidth = torsoRef * 0.05f + 1f
        }

        // ---- 四肢胶囊体（粗线 + 圆头） ----
        val limbs = listOf(
            Triple(PoseModel.SHOULDER_L, PoseModel.ELBOW_L, 0.30f),
            Triple(PoseModel.ELBOW_L, PoseModel.WRIST_L, 0.26f),
            Triple(PoseModel.SHOULDER_R, PoseModel.ELBOW_R, 0.30f),
            Triple(PoseModel.ELBOW_R, PoseModel.WRIST_R, 0.26f),
            Triple(PoseModel.HIP_L, PoseModel.ANKLE_L, 0.36f),
            Triple(PoseModel.HIP_R, PoseModel.ANKLE_R, 0.36f),
        )
        limbs.forEachIndexed { idx, (a, b, thick) ->
            val paint = if (perBoneColors != null && idx < perBoneColors.size) {
                Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    color = perBoneColors[idx]
                    style = Paint.Style.FILL
                }
            } else fillPaint
            drawCapsule(canvas, pose, a, b, torsoRef * thick, paint, strokePaint, perBoneColors == null)
        }

        // ---- 躯干：肩线 + 髋线构成的圆角梯形 ----
        drawTorso(canvas, pose, torsoRef, fillPaint, strokePaint, perBoneColors == null)

        // ---- 头部圆 ----
        val headR = torsoRef * 0.46f
        canvas.drawCircle(pose.jx(PoseModel.HEAD), pose.jy(PoseModel.HEAD), headR, fillPaint)
        if (perBoneColors == null) {
            canvas.drawCircle(pose.jx(PoseModel.HEAD), pose.jy(PoseModel.HEAD), headR, strokePaint)
        }
    }

    /** 胶囊体：两关节点之间的粗圆头线段 */
    private fun drawCapsule(
        canvas: Canvas, pose: FloatArray, a: Int, b: Int, width: Float,
        fill: Paint, stroke: Paint, withStroke: Boolean,
    ) {
        val p = Paint(fill).apply {
            style = Paint.Style.STROKE
            strokeWidth = width
            strokeCap = Paint.Cap.ROUND
        }
        canvas.drawLine(pose.jx(a), pose.jy(a), pose.jx(b), pose.jy(b), p)
        if (withStroke) {
            // 描边：比填充略粗的先画一层
            val sp = Paint(stroke).apply {
                style = Paint.Style.STROKE
                strokeCap = Paint.Cap.ROUND
            }
            // 先描边（粗）再填充（细）才能看到边——重画顺序
            sp.strokeWidth = width * 1.18f
            p.strokeWidth = width
            canvas.drawLine(pose.jx(a), pose.jy(a), pose.jx(b), pose.jy(b), sp)
            canvas.drawLine(pose.jx(a), pose.jy(a), pose.jx(b), pose.jy(b), p)
        }
    }

    /** 躯干圆角梯形：肩线两端外扩、髋线两端外扩后连接 */
    private fun drawTorso(
        canvas: Canvas, pose: FloatArray, torsoRef: Float,
        fill: Paint, stroke: Paint, withStroke: Boolean,
    ) {
        val sl = PoseModel.SHOULDER_L; val sr = PoseModel.SHOULDER_R
        val hl = PoseModel.HIP_L; val hr = PoseModel.HIP_R
        val expand = 0.22f * torsoRef

        fun out(a: Int, b: Int, t: Float): FloatArray {
            val dx = pose.jx(b) - pose.jx(a); val dy = pose.jy(b) - pose.jy(a)
            return floatArrayOf(pose.jx(a) - dx * t, pose.jy(a) - dy * t, pose.jx(b) + dx * t, pose.jy(b) + dy * t)
        }

        val shoulder = out(sl, sr, expand / (distOf(pose, sl, sr) + 1e-4f))
        val hip = out(hl, hr, expand / (distOf(pose, hl, hr) + 1e-4f))

        val path = Path().apply {
            moveTo(shoulder[0], shoulder[1])
            lineTo(shoulder[2], shoulder[3])
            lineTo(hip[2], hip[3])
            lineTo(hip[0], hip[1])
            close()
        }
        canvas.drawPath(path, fill)
        if (withStroke) canvas.drawPath(path, stroke)

        // 颈：头与躯干之间的连接柱
        val neckPaint = Paint(fill).apply {
            style = Paint.Style.STROKE
            strokeWidth = torsoRef * 0.24f
            strokeCap = Paint.Cap.ROUND
        }
        canvas.drawLine(
            pose.jx(PoseModel.NECK), pose.jy(PoseModel.NECK),
            pose.jx(PoseModel.CHEST), pose.jy(PoseModel.CHEST), neckPaint
        )
    }

    private fun distOf(p: FloatArray, a: Int, b: Int): Float {
        val dx = p.jx(b) - p.jx(a); val dy = p.jy(b) - p.jy(a)
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }

    /** 渲染姿势缩略图（供 AI 合成姿势的卡片显示） */
    fun renderThumbnail(pose75: FloatArray, sizePx: Int, palette: Palette, background: Int): Bitmap {
        val bmp = Bitmap.createBitmap(sizePx, sizePx, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        canvas.drawColor(background)

        // 归一化坐标 → 缩略图坐标：以主体关节为中心自适应取景
        val keyIdx = intArrayOf(0, 3, 4, 5, 10, 11, 13, 14, 16, 17, 12, 15, 7, 8)
        var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
        var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
        for (i in keyIdx) {
            minX = minOf(minX, pose75.jx(i)); maxX = maxOf(maxX, pose75.jx(i))
            minY = minOf(minY, pose75.jy(i)); maxY = maxOf(maxY, pose75.jy(i))
        }
        val cx = (minX + maxX) / 2; val cy = (minY + maxY) / 2
        val span = maxOf(maxX - minX, maxY - minY) * 1.25f + 1e-4f
        val scale = sizePx / span

        val px = FloatArray(PoseModel.POSE_DIM)
        for (i in 0 until PoseModel.NUM_JOINTS) {
            px[i * 3] = (pose75.jx(i) - cx) * scale + sizePx / 2f
            px[i * 3 + 1] = (pose75.jy(i) - cy) * scale + sizePx / 2f
            px[i * 3 + 2] = pose75[i * 3 + 2]
        }
        val torso = distOf(px, PoseModel.PELVIS, PoseModel.NECK)
        draw(canvas, px, torso, palette)
        return bmp
    }
}
