import colored as cl
import numpy as np
import torch
from scipy.signal import find_peaks


def find_peaks_and_valleys(filtered_wave):
    peaks, _ = find_peaks(filtered_wave - filtered_wave.mean())
    valleys, _ = find_peaks(-filtered_wave + filtered_wave.mean())
    return peaks, valleys


def find_steps(filtered_wave, tolerance=30):
    peaks, valleys = find_peaks_and_valleys(filtered_wave)

    while np.min(np.diff(peaks)) < tolerance:
        # print(len(peaks), np.min(np.diff(peaks)))
        temp = []
        i = 0
        while i < len(peaks) - 1:
            if peaks[i + 1] - peaks[i] < tolerance:
                temp.append(
                    peaks[i + 1]
                    if filtered_wave[peaks[i + 1]] > filtered_wave[peaks[i]]
                    else peaks[i]
                )
                i += 1
            else:
                temp.append(peaks[i])
            i += 1
        peaks = np.array(temp)

    while np.min(np.diff(valleys)) < tolerance:
        temp = []
        i = 0
        while i < len(valleys) - 1:
            if valleys[i + 1] - valleys[i] < tolerance:
                temp.append(
                    valleys[i + 1]
                    if filtered_wave[valleys[i + 1]] < filtered_wave[valleys[i]]
                    else valleys[i]
                )
                i += 1
            else:
                temp.append(valleys[i])
            i += 1
        valleys = np.array(temp)

    # if there exist two valleys between two peaks, keep the lowest valley
    for i in range(1, len(peaks)):
        temp = valleys[(valleys > peaks[i - 1]) & (valleys < peaks[i])]
        while len(temp) > 1:
            valleys = valleys[valleys != temp.max()]
            # remove the max
            temp = temp[temp != temp.max()]

    # vice versa
    for i in range(1, len(valleys)):
        temp = peaks[(peaks > valleys[i - 1]) & (peaks < valleys[i])]
        while len(temp) > 1:
            peaks = peaks[peaks != temp.min()]
            temp = temp[temp != temp.min()]

    # append the missing peaks and valleys
    missing_valleys = []
    for i in range(1, len(peaks)):
        temp = valleys[(valleys > peaks[i - 1]) & (valleys < peaks[i])]
        if len(temp) == 0:
            missing_valleys.append(i)
    for i in missing_valleys:
        new_valley = np.argmin(filtered_wave[peaks[i - 1] : peaks[i]]) + peaks[i - 1]
        insert_idx = np.searchsorted(valleys, new_valley)
        valleys = np.insert(valleys, insert_idx, new_valley)

    missing_peaks = []
    for i in range(1, len(valleys)):
        temp = peaks[(peaks > valleys[i - 1]) & (peaks < valleys[i])]
        if len(temp) == 0:
            missing_peaks.append(i)
    for i in missing_peaks:
        new_peak = (
            np.argmax(filtered_wave[valleys[i - 1] : valleys[i]]) + valleys[i - 1]
        )
        insert_idx = np.searchsorted(peaks, new_peak)
        peaks = np.insert(peaks, insert_idx, new_peak)

    if len(peaks) != len(valleys):
        # print(
        #     cl.Fore.yellow
        #     + f"Peaks and valleys do not match {len(peaks)} {len(valleys)}",
        #     cl.Style.reset,
        # )
        # drop the last one of the longer list
        if len(peaks) > len(valleys):
            peaks = peaks[:-1]
        else:
            valleys = valleys[:-1]
    # assert len(peaks) == len(
    #     valleys
    # ), f"Peaks and valleys do not match {len(peaks)} {len(valleys)}"

    # check the peaks is strictly increasing
    assert np.all(np.diff(peaks) > 0), "Peaks are not strictly increasing"
    assert np.all(np.diff(valleys) > 0), "Valleys are not strictly increasing"
    # check any redundant peaks or valleys
    assert len(peaks) == len(np.unique(peaks)), "Redundant peaks found"
    assert len(valleys) == len(np.unique(valleys)), "Redundant valleys found"

    return peaks, valleys


# from typing import Union

# import numpy as np
# import torch
# from tqdm import tqdm

# # make a function of step detection
# # ilterate over the acc vairable
# # if the acc is greater than 1.5 then it is a step
# # then grab 100 data points before and after the step


# def stepDetection(acc: np.array) -> np.array:
#     shift = acc.std() * 0.5
#     UPTHRESHOLD = acc.mean() + shift
#     DOWNTHRESHOLD = acc.mean() - shift
#     stepStart = []
#     stepEnd = []
#     reachTop = False
#     for i in range(len(acc)):
#         if acc[i] > UPTHRESHOLD and reachTop == False:
#             stepStart.append(i)
#             reachTop = True

#         # elif acc[i] < 8 and reachTop == True:
#         #     reachTop = False
#         #     # remove the stepStart last element
#         #     stepStart.pop()

#         elif acc[i] < DOWNTHRESHOLD and reachTop == True:
#             stepEnd.append(i)
#             reachTop = False
#             # reverse trace the stepStart
#             stepStart.pop()
#             reverseTop = False
#             for j in range(i, i - 100, -1):
#                 # print(j,acc[j], reverseTop)
#                 if acc[j] > UPTHRESHOLD and reverseTop == False:
#                     reverseTop = True
#                 elif acc[j] < UPTHRESHOLD and reverseTop == True:
#                     reverseTop = False
#                     stepStart.append(j)
#                     break

#             if len(stepStart) != len(stepEnd):
#                 # remove start last element
#                 stepEnd.pop()
#             assert len(stepStart) == len(stepEnd)
#             # if add step start and end distance is larger than 100 then remove the last step
#             if len(stepStart) != 0:
#                 if stepEnd[-1] - stepStart[-1] >= 100:
#                     stepStart.pop()
#                     stepEnd.pop()

#     if len(stepStart) != len(stepEnd):
#         # remove start last element
#         stepStart.pop()

#     assert len(stepStart) == len(stepEnd)
#     # print("Number of steps: ", len(stepStart))
#     return np.array(stepStart), np.array(stepEnd)


# def endStepBackwardSampling(
#     acc: Union[np.array, torch.tensor],
#     gyr: Union[np.array, torch.tensor],
#     mag: Union[np.array, torch.tensor],
#     stepEnd: Union[np.array, torch.tensor],
#     WINDOW_SIZE: int = 100,
# ) -> Union[np.array, torch.tensor]:
#     # check tensor or numpy
#     if isinstance(acc, torch.Tensor):
#         is_torch = True
#     else:
#         is_torch = False
#     if not is_torch:
#         acc = torch.from_numpy(acc).float()
#         gyr = torch.from_numpy(gyr).float()
#         mag = torch.from_numpy(mag).float()
#         stepEnd = torch.from_numpy(stepEnd).float()

#     # check the shape
#     assert acc.shape[-1] == 4
#     assert gyr.shape[-1] == 4
#     assert mag.shape[-1] == 4
#     assert acc.shape[0] == gyr.shape[0] == mag.shape[0]

#     raw = torch.cat((acc, gyr, mag), dim=1)

#     data = []
#     for index in stepEnd:
#         if index - WINDOW_SIZE < 0:
#             continue
#         newdata = raw[index - WINDOW_SIZE : index + WINDOW_SIZE].swapaxes(0, 1)
#         if not is_torch:
#             newdata = newdata.numpy()
#         data.append(newdata)

#     return torch.stack(data) if is_torch else np.stack(data)


# def startStepForwardSampling(
#     acc: Union[np.array, torch.tensor],
#     gyr: Union[np.array, torch.tensor],
#     mag: Union[np.array, torch.tensor],
#     stepStart: Union[np.array, torch.tensor],
#     WINDOW_SIZE: int = 100,
# ) -> Union[np.array, torch.tensor]:
#     # check tensor or numpy
#     if isinstance(acc, torch.Tensor):
#         is_torch = True
#     else:
#         is_torch = False
#     if not is_torch:
#         acc = torch.from_numpy(acc).float()
#         gyr = torch.from_numpy(gyr).float()
#         mag = torch.from_numpy(mag).float()
#         stepStart = torch.from_numpy(stepStart).float()

#     # check the shape
#     assert acc.shape[-1] == 4
#     assert gyr.shape[-1] == 4
#     assert mag.shape[-1] == 4
#     assert acc.shape[0] == gyr.shape[0] == mag.shape[0]

#     raw = torch.cat((acc, gyr, mag), dim=1)

#     data = []
#     for index in stepStart:
#         if index + WINDOW_SIZE > len(raw):
#             continue
#         newdata = raw[index - WINDOW_SIZE : index + WINDOW_SIZE].swapaxes(0, 1)
#         if not is_torch:
#             newdata = newdata.numpy()
#         data.append(newdata)

#     return torch.stack(data) if is_torch else np.stack(data)
