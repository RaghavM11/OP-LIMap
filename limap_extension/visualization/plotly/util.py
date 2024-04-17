FRAME_DURATION_MS_DEFAULT = 50  # [ms]

TRACKING_MARKER_SIZE = 5
DEFAULT_TRACKING_COLOR = 'red'

POINT_CLOUD_MARKER_SIZE = 3
POINT_CLOUD_OPACITY = 0.4
DEFAULT_POINT_CLOUD_COLOR = 'royalblue'

MESH_OPACITY_DEFAULT = 0.75


def make_update_menus(frame_duration_ms: float = FRAME_DURATION_MS_DEFAULT):
    updatemenus = [
        dict(type="buttons",
             buttons=[
                 dict(label="Play",
                      method="animate",
                      args=[
                          None,
                          dict(frame=dict(duration=frame_duration_ms, redraw=True),
                               fromcurrent=True,
                               transition=dict(duration=frame_duration_ms / 2,
                                               easing="quadratic-in-out"))
                      ]),
                 dict(label="Pause",
                      method="animate",
                      args=[[None],
                            dict(frame=dict(duration=0, redraw=False),
                                 mode="immediate",
                                 transition=dict(duration=0))])
             ],
             showactive=True)
    ]
    return updatemenus


def make_sliders(num_frames: int, frame_duration_ms: float = FRAME_DURATION_MS_DEFAULT):
    sliders = dict(
        active=0,
        currentvalue=dict(
            font=dict(size=20),
            prefix="Step: ",
            visible=True,
            xanchor="left"  # defines where the description of current value is along slider bar.
        ),
        transition=dict(duration=frame_duration_ms / 2, easing="cubic-in-out"),
        steps=[])
    for i in range(num_frames):
        slider_step = dict(args=[[i],
                                 dict(frame=dict(duration=frame_duration_ms / 2, redraw=True),
                                      mode="immediate",
                                      transition=dict(duration=frame_duration_ms / 2))],
                           label=i,
                           method="animate")
        sliders["steps"].append(slider_step)
    return sliders