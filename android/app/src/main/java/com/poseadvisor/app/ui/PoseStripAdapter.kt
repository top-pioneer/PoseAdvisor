package com.poseadvisor.app.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.poseadvisor.app.R
import com.poseadvisor.app.pose.PoseEntry

/** 底部姿势推荐卡条（支持库姿势与 AI 合成姿势混合展示） */
class PoseStripAdapter(
    private val onSelect: (PoseEntry) -> Unit,
) : RecyclerView.Adapter<PoseStripAdapter.VH>() {

    /** 卡片条目：库姿势（assets 缩略图）或 AI 合成姿势（内存 Bitmap） */
    data class Item(
        val entry: PoseEntry,
        val synthesizedBitmap: Bitmap? = null,
    )

    private var items: List<Item> = emptyList()
    var selectedId: String? = null
        private set

    fun submit(list: List<Item>) {
        items = list
        selectedId = items.firstOrNull()?.entry?.id
        notifyDataSetChanged()
    }

    fun current(): PoseEntry? = items.firstOrNull { it.entry.id == selectedId }?.entry

    fun indexOfCurrent(): Int = items.indexOfFirst { it.entry.id == selectedId }

    /** 智能推荐自动轮换时同步卡片选中态（不触发点击回调） */
    fun selectById(id: String) {
        val old = selectedId
        if (old == id) return
        selectedId = id
        notifyItemChanged(items.indexOfFirst { it.entry.id == old })
        notifyItemChanged(items.indexOfFirst { it.entry.id == id })
    }

    class VH(view: View) : RecyclerView.ViewHolder(view) {
        val thumb: ImageView = view.findViewById(R.id.poseThumb)
        val label: TextView = view.findViewById(R.id.poseLabel)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val v = LayoutInflater.from(parent.context).inflate(R.layout.item_pose, parent, false)
        return VH(v)
    }

    override fun getItemCount() = items.size

    override fun onBindViewHolder(holder: VH, position: Int) {
        val item = items[position]
        holder.label.text = item.entry.label
        if (item.synthesizedBitmap != null) {
            holder.thumb.setImageBitmap(item.synthesizedBitmap)
        } else {
            try {
                holder.itemView.context.assets.open(item.entry.thumbnail).use { input ->
                    holder.thumb.setImageBitmap(BitmapFactory.decodeStream(input))
                }
            } catch (_: Exception) {
                holder.thumb.setImageDrawable(null)
            }
        }
        val selected = item.entry.id == selectedId
        holder.thumb.setBackgroundResource(
            if (selected) R.drawable.bg_pose_card_selected else R.drawable.bg_pose_card
        )
        holder.itemView.setOnClickListener {
            val old = selectedId
            selectedId = item.entry.id
            notifyItemChanged(items.indexOfFirst { it.entry.id == old })
            notifyItemChanged(position)
            onSelect(item.entry)
        }
    }
}
