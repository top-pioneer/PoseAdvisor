package com.poseadvisor.app.ui

import android.graphics.BitmapFactory
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.poseadvisor.app.R
import com.poseadvisor.app.pose.PoseEntry

/** 底部姿势推荐卡条 */
class PoseStripAdapter(
    private val onSelect: (PoseEntry) -> Unit,
) : RecyclerView.Adapter<PoseStripAdapter.VH>() {

    private var items: List<PoseEntry> = emptyList()
    var selectedId: String? = null
        private set

    fun submit(list: List<PoseEntry>) {
        items = list
        selectedId = list.firstOrNull()?.id
        notifyDataSetChanged()
    }

    fun current(): PoseEntry? = items.firstOrNull { it.id == selectedId }

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
        holder.label.text = item.label
        try {
            holder.itemView.context.assets.open(item.thumbnail).use { input ->
                holder.thumb.setImageBitmap(BitmapFactory.decodeStream(input))
            }
        } catch (_: Exception) {
            holder.thumb.setImageDrawable(null)
        }
        val selected = item.id == selectedId
        holder.thumb.setBackgroundResource(
            if (selected) R.drawable.bg_pose_card_selected else R.drawable.bg_pose_card
        )
        holder.itemView.setOnClickListener {
            val old = selectedId
            selectedId = item.id
            notifyItemChanged(items.indexOfFirst { it.id == old })
            notifyItemChanged(position)
            onSelect(item)
        }
    }
}
