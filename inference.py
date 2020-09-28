import click
import pandas as pd
import os
import cv2
import numpy as np
import models
import utils
import re


def mass_center(mask):
    # calculate mass center from top-left corner
    x_by_mass = 0
    y_by_mass = 0
    total_mass = np.sum(mask)
    for x in np.arange(0, mask.shape[0]):
        x_by_mass += np.sum(x * mask[:, x])
        y_by_mass += np.sum(x * mask[x, :])

    return (x_by_mass / total_mass, y_by_mass / total_mass)


def connected_components_with_threshold(image, threshold, ground_truth):
    """
    Function that takes a mask and filters its component given a provided threshold
    this returns the number of resulting components and a new filtered mask (tuple)
    """

    num_components, mask = cv2.connectedComponents(image)
    filtered_mask = np.zeros_like(image, dtype=np.uint8)
    component_list = []
    mass_center_array = []
    iou_array = []

    for component in np.arange(1, num_components):
        isolated_component = mask == component
        if np.sum(isolated_component) >= threshold:
            iou_array.append(
                np.sum(np.logical_and(isolated_component, ground_truth))
                / np.sum(np.logical_or(isolated_component, ground_truth))
            )
            mass_center_array.append(
                [component, mass_center(isolated_component.astype(int))]
            )
            filtered_mask += isolated_component.astype(np.uint8) * component
            component_list.append(component)
    if len(component_list) == 0:
        mass_center_array = None
        return (
            len(component_list),
            filtered_mask,
            mass_center_array,
            np.asarray(iou_array),
        )
    else:
        return (
            len(component_list),
            filtered_mask,
            np.asarray(mass_center_array, dtype=object),
            np.asarray(iou_array),
        )


@click.command()
@click.option(
    "--model", required=True, help="Model name in output/models folder e.g model.h5"
)
@click.option(
    "--output",
    default=os.path.join(".", "output", "validation"),
    help="Path to which the generated masks are written",
)
@click.option(
    "--csv",
    default=os.path.join(".", "test.csv"),
    help="Test set csv with (image, mask) tuples",
)
@click.option("--imgpath", default=os.path.join(".", "images"), help="Base images path")
@click.option(
    "--valid", default=False, is_flag=True, help="Generate validation csv or not"
)
def inference(model, output, csv, imgpath, valid):
    np.random.seed(0)
    test_set_full = pd.read_csv(csv)
    list_ids = list(test_set_full["image"].values)
    list_masks = list(test_set_full["mask"].values)
    # get root directories
    base_img_path = imgpath
    img_path = os.path.join(base_img_path, "images")
    mask_path = os.path.join(base_img_path, "masks")

    list_ids = list_ids
    list_masks = list_masks
    labels = dict(zip(list_ids, list_masks))

    if valid:
        ground_truth_csv = pd.read_csv(
            os.path.join("single_instance_dataset_wradius.csv")
        )

    mask_output_path = os.path.join(output, model, "prediction_masks")
    if not os.path.exists(mask_output_path):
        os.makedirs(mask_output_path)

    valid_generator = utils.DataGeneratorMobileNetKeras(
        batch_size=1,
        img_path=img_path,
        labels=labels,
        list_IDs=list_ids,
        n_channels=3,
        n_channels_label=1,
        shuffle=False,
        mask_path=mask_path,
        augmentation=False,
    )

    model_loaded = models.load_model(os.path.join(".", "output", "models", model))
    prediction = model_loaded.predict_generator(
        generator=valid_generator, use_multiprocessing=True, verbose=True
    )

    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    test_images = list_ids

    valid_list = list()
    nan_keys = [
        "component_x",
        "component_y",
        "gt_area",
        "pixel_distance",
        "norm_distance",
        "prediction_area",
        "component_area",
        "component_relative_area",
        "component_intersection",
        "component_union",
        "component_iou",
        "component_pw_recall",
        "component_pw_precision",
    ]

    for threshold in threshold_list:
        array_pred = np.copy(prediction)
        for i in np.arange(0, prediction.shape[0]):
            # get prediction and normalize
            pred = array_pred[i, :, :, 0]
            pred = (pred > threshold).astype(bool)
            cv2.imwrite(
                mask_output_path + "/" + (str(threshold)) + test_images[i],
                (pred.astype(np.uint8)) * 255,
            )
            if valid:
                data = {}
                row = utils.get_sample_ground_truth(test_images[i], ground_truth_csv)
                gt_center = np.ndarray([1, 2])
                gt_center[0, 0] = (row["x_center_resize"].values[0]) / 2
                gt_center[0, 1] = (row["y_center_resize"].values[0]) / 2
                diam_resize = (row["diam_resize"].values[0]) / 2

                ground_truth = utils.read_image_grayscale(
                    os.path.join(mask_path, "mask_" + test_images[i][:-3] + "png")
                )
                ground_truth = cv2.resize(ground_truth, (0, 0), fx=0.5, fy=0.5)
                ground_truth = (ground_truth > 0).astype(int)

                (
                    num_labels,
                    labeled_img,
                    centers,
                    iou_array,
                ) = connected_components_with_threshold(
                    (pred).astype(np.uint8), 0, ground_truth
                )

                gt_area = np.sum(ground_truth)
                prediction_area = np.sum(pred)
                # computo de metricas segun resultado de component analysis

                if centers is not None:
                    for component_label in range(num_labels):
                        filtered_component = labeled_img == (component_label + 1)
                        component_center = mass_center(filtered_component)

                        pixel_distance = np.linalg.norm(
                            np.subtract(gt_center, component_center)
                        )

                        data["mask_name"] = test_images[i]
                        data["threshold"] = str(threshold)
                        data["mask_x"] = gt_center[0, 0]
                        data["mask_y"] = gt_center[0, 1]
                        # component_x
                        data["component_x"] = component_center[0]
                        # component_y
                        data["component_y"] = component_center[1]
                        data["pixel_distance"] = pixel_distance
                        data["norm_distance"] = pixel_distance / diam_resize
                        data["gt_area"] = gt_area
                        data["prediction_area"] = prediction_area
                        # component_area
                        component_area = np.sum(filtered_component)
                        data["component_area"] = component_area
                        # relative_area
                        data["component_relative_area"] = (
                            component_area / prediction_area
                        )
                        # component pixelwise metrics
                        intersection = np.sum(
                            np.logical_and(filtered_component, ground_truth)
                        )
                        union = np.sum(np.logical_or(filtered_component, ground_truth))
                        data["component_intersection"] = intersection
                        data["component_union"] = union
                        data["component_iou"] = intersection / union
                        data["component_pw_recall"] = intersection / gt_area
                        data["component_pw_precision"] = intersection / component_area

                else:  # no buds detected register it in the metrics dict
                    data["mask_name"] = test_images[i]
                    data["threshold"] = str(threshold)
                    data["mask_x"] = gt_center[0, 0]
                    data["mask_y"] = gt_center[0, 1]
                    # mass assing np.nan to nan-able keys
                    data.update(dict.fromkeys(nan_keys, np.nan))

                valid_list.append(data)

    print(model + " images generated!")
    if valid:
        csv = pd.DataFrame(valid_list)
        csv.to_csv(os.path.join(output, model[:-3] + ".csv"))


if __name__ == "__main__":
    inference()