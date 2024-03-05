import numpy as np

data = np.load('bounding_box_3d_0000.npy')
print(data[0].dtype)
for bbox_2d in data:
    id = bbox_2d["semanticId"]
    x_min = bbox_2d["x_min"]
    y_min = bbox_2d["y_min"]
    z_min = bbox_2d["z_min"]
    x_max = bbox_2d["x_max"]
    y_max = bbox_2d["y_max"]
    z_max = bbox_2d["z_max"]
    print(f'x [{x_min}-> {x_max}]')
    print(f'y [{y_min}-> {y_max}]')
    print(f'z [{z_min}-> {z_max}]')
    break
    # color = data_to_colour(id)
    # labels = id_to_labels[str(id)]
    # rect = patches.Rectangle(
    #     xy=(bbox_2d["x_min"], bbox_2d["y_min"]),
    #     width=bbox_2d["x_max"] - bbox_2d["x_min"],
    #     height=bbox_2d["y_max"] - bbox_2d["y_min"],
    #     edgecolor=color,
    #     linewidth=2,
    #     label=labels,
    #     fill=False,
    # )
    # ax.add_patch(rect)
