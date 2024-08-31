import matplotlib.pyplot as plt

def visualize_bboxes(image, bboxes, labels=None):
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    for i, bbox in enumerate(bboxes):
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                             fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
        if labels is not None:
            plt.text(bbox[0], bbox[1] - 2, f'{labels[i]}', bbox=dict(facecolor='blue', alpha=0.5), fontsize=12, color='white')
    plt.axis('off')
    plt.show()
