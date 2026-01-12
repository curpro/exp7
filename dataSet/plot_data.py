import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
import os
import glob


def plot_with_camera_tool(dataset_root):
    # ---------------------------------------------------------
    # 1. 扫描数据
    # ---------------------------------------------------------
    search_pattern = os.path.join(dataset_root, 'processed_data*')
    target_dirs = [d for d in glob.glob(search_pattern) if os.path.isdir(d)]
    target_dirs.sort()

    all_files = []
    for d in target_dirs:
        files = glob.glob(os.path.join(d, '*.csv'))
        all_files.extend(files)

    if not all_files:
        print(f"未在 {dataset_root} 下找到 CSV 文件。")
        return

    print(f"共找到 {len(all_files)} 条轨迹，正在绘制...")

    fig = go.Figure()

    # 颜色池
    color_palette = pc.qualitative.Dark24 + pc.qualitative.Alphabet + pc.qualitative.Light24

    for i, file_path in enumerate(all_files):
        try:
            df = pd.read_csv(file_path)
            if not {'x', 'y', 'z'}.issubset(df.columns):
                continue

            trace_name = f"{os.path.basename(os.path.dirname(file_path))}/{os.path.basename(file_path)}"
            this_color = color_palette[i % len(color_palette)]

            # 速度数据 (仅用于Hover)
            if {'vx', 'vy', 'vz'}.issubset(df.columns):
                speed_val = (df['vx'] ** 2 + df['vy'] ** 2 + df['vz'] ** 2) ** 0.5
                hover_template = f"<b>{trace_name}</b><br>Speed: %{{text:.1f}} m/s<extra></extra>"
            else:
                speed_val = ["N/A"] * len(df)
                hover_template = f"<b>{trace_name}</b><extra></extra>"

            # 1. 轨迹线
            fig.add_trace(go.Scatter3d(
                x=df['x'], y=df['y'], z=df['z'],
                mode='lines',
                name=trace_name,
                line=dict(color=this_color, width=4),
                text=speed_val,
                hovertemplate=hover_template
            ))

            # 2. 地面投影
            z_min = df['z'].min()
            offset = (df['z'].max() - z_min) * 0.1
            proj_z = z_min - (offset if offset > 10 else 10)

            fig.add_trace(go.Scatter3d(
                x=df['x'], y=df['y'], z=[proj_z] * len(df),
                mode='lines',
                line=dict(color='gray', width=2),
                opacity=0.15,
                hoverinfo='skip'
            ))

            # 3. 起终点
            # fig.add_trace(go.Scatter3d(
            #     x=[df['x'].iloc[0]], y=[df['y'].iloc[0]], z=[df['z'].iloc[0]],
            #     mode='markers',
            #     marker=dict(size=4, color='#00FF00', symbol='diamond'),
            #     hoverinfo='skip'
            # ))
            # fig.add_trace(go.Scatter3d(
            #     x=[df['x'].iloc[-1]], y=[df['y'].iloc[-1]], z=[df['z'].iloc[-1]],
            #     mode='markers',
            #     marker=dict(size=4, color='#FF0000', symbol='x'),
            #     hoverinfo='skip'
            # ))

        except Exception as e:
            print(f"跳过 {file_path}: {e}")

    # ---------------------------------------------------------
    # 布局设置
    # ---------------------------------------------------------
    fig.update_layout(
        template='plotly_white',
        width=1800,
        height=1800,
        margin=dict(r=0, l=0, b=0, t=0),
        showlegend=False,  # 隐藏左侧列表
        scene=dict(
            xaxis=dict(title='X (East) (m)', gridcolor='lightgray', showbackground=False),
            yaxis=dict(title='Y (North) (m)', gridcolor='lightgray', showbackground=False),
            zaxis=dict(title='Z (Altitude) (m)', gridcolor='lightgray', showbackground=False),
            aspectmode='data'
        )
    )

    # ---------------------------------------------------------
    # 关键修改：配置相机工具栏 (Config)
    # ---------------------------------------------------------
    config = {
        'displayModeBar': True,  # 强制显示工具栏
        'scrollZoom': True,
        'displaylogo': False,  # 隐藏 Plotly 的 logo
        'toImageButtonOptions': {
            'format': 'png',  # 下载格式
            'filename': 'trajectory_screenshot',
            'height': 2160,  # 高度 (4K)
            'width': 3840,  # 宽度 (4K)
            'scale': 1  # 缩放比例
        }
    }

    output_html = 'all_plot_trajectory.html'
    # 将 config 传入 write_html
    fig.write_html(output_html, config=config)
    print(f"\n✅ 已生成带相机功能的图表: {output_html}")
    print("现在您把鼠标移动到图表右上角，应该能看到相机图标，点击即可下载高清截图。")


if __name__ == "__main__":
    plot_with_camera_tool('../dataSet')