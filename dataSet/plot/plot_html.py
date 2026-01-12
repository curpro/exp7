import pandas as pd
import plotly.graph_objects as go
import os


def plot_interactive_HD(file_path):
    if not os.path.exists(file_path):
        # 尝试在当前目录下查找
        local_path = os.path.basename(file_path)
        if os.path.exists(local_path):
            file_path = local_path
        else:
            print(f"错误：找不到文件 {file_path}")
            return

    print(f"读取数据中: {file_path}")
    df = pd.read_csv(file_path)

    # ---------------------------------------------------------
    # 核心优化：创建更清晰的 3D 轨迹
    # ---------------------------------------------------------
    fig = go.Figure()

    # 1. 绘制主轨迹线
    df['speed'] = (df['vx'] ** 2 + df['vy'] ** 2 + df['vz'] ** 2) ** 0.5
    speed_col = 'speed'

    # 1. 绘制主轨迹线 (颜色代表速度)
    fig.add_trace(go.Scatter3d(
        x=df['x'],
        y=df['y'],
        z=df['z'],
        mode='lines',
        name='飞行轨迹',
        showlegend=False,
        line=dict(
            color=df[speed_col],  # <--- 这里改成了速度
            colorscale='Turbo',  # 这种色谱红蓝变化明显，适合看速度
            width=6,
            showscale=True,
            colorbar=dict(
                title='速度(m/s)',  # <--- 标题已改
                x=0.95,  # 如果你之前改过布局，这里可以按需调整(比如0.8)
                y=0.5,
                len=0.6,
                thickness=15
            )
        ),
        # 修改鼠标悬停提示，增加 Speed 显示
        hovertemplate=(
            '<b>X</b>: %{x:.1f}m<br>'
            '<b>Y</b>: %{y:.1f}m<br>'
            '<b>Z</b>: %{z:.1f}m<br>'
            '<b>Speed</b>: %{text:.1f}<extra></extra>'
        ),
        text=df[speed_col]  # 把速度值传进去
    ))

    # 2. 绘制“地面投影” (增强立体感)
    z_min = df['z'].min()
    # 为了防止投影与轨迹太远或太近，设置投影面为最低点下方 10% 范围
    offset = (df['z'].max() - z_min) * 0.1
    proj_z = z_min - (offset if offset > 10 else 10)

    fig.add_trace(go.Scatter3d(
        x=df['x'],
        y=df['y'],
        z=[proj_z] * len(df),
        mode='lines',
        name='地面投影',
        line=dict(color='gray', width=3),
        opacity=0.3,
        hoverinfo='skip'
    ))

    # 3. 标记起点
    fig.add_trace(go.Scatter3d(
        x=[df['x'].iloc[0]], y=[df['y'].iloc[0]], z=[df['z'].iloc[0]],
        mode='markers+text',
        name='起点',
        marker=dict(size=10, color='#00FF00', symbol='diamond'),
        text=["START"],
        textposition="top center",
        textfont=dict(color='#00FF00', size=12, family="Arial Black")
    ))

    # 4. 标记终点
    fig.add_trace(go.Scatter3d(
        x=[df['x'].iloc[-1]], y=[df['y'].iloc[-1]], z=[df['z'].iloc[-1]],
        mode='markers+text',
        name='终点',
        marker=dict(size=10, color='#FF0000', symbol='x'),
        text=["END"],
        textposition="top center",
        textfont=dict(color='#FF0000', size=12, family="Arial Black")
    ))

    # ---------------------------------------------------------
    # 布局优化
    # ---------------------------------------------------------
    fig.update_layout(
        title={
            'text': f"F-16 机动轨迹可视化",
            'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=24, color='black')
        },
        template='plotly_white',
        width=1200,
        height=800,
        # 1. 压缩边缘，不留白
        margin=dict(r=5, l=5, b=50, t=40),

        # 2. 关键修改：把图例搬进画面里
        legend=dict(
            yanchor="top",
            y=0.9,  # 垂直位置：0.9 代表在顶部偏下一点
            xanchor="right",
            x=0.85,  # 水平位置：0.95 代表在画面最右侧内部
            bgcolor="rgba(255, 255, 255, 0.6)",  # 给图例加个半透明白底，防止看不清
            bordercolor="lightgray",
            borderwidth=1,
            itemsizing='constant'  # 统一图标大小
        ),

        scene=dict(
            xaxis=dict(title='X (East) (m)', gridcolor='lightgray', showbackground=False),
            yaxis=dict(title='Y (North) (m)', gridcolor='lightgray', showbackground=False),
            zaxis=dict(title='Z (Altitude) (m)', gridcolor='lightgray', showbackground=False),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.1, y=1.1, z=0.4),  # 保持拉近的视角
                center=dict(x=0, y=0, z=-0.1)
            )
        )
    )

    # 输出文件名为 _a_3d_HD.html
    output_html = file_path.replace('.csv', '.html')
    # 如果路径中包含目录，确保写入权限，或者直接写到当前目录
    if '\\' in output_html or '/' in output_html:
        output_html = os.path.basename(output_html)

    fig.write_html(output_html)
    print(f"✅ 高清图表已生成: {output_html}")

    config = {
        'toImageButtonOptions': {
            'format': 'png',  # 图片格式
            'filename': 'my_flight_trace',  # 下载的文件名
            'height': 1080,  # 图片高度
            'width': 1920,  # 图片宽度 (1920x1080 为 1080p)
            'scale': 2  # 放大倍数 (2 表示 3840x2160, 即 4K 超清)
        }
    }
    try:
        fig.show(config=config)
    except:
        pass


if __name__ == "__main__":
    # --- 修改此处：适配你的新数据集路径 ---
    csv_path = r'D:\AFS\lunwen\dataSet\zreal\12Roll left.csv'

    # 本地测试备用:
    # csv_path = 'f16_super_maneuver_a.csv'

    plot_interactive_HD(csv_path)