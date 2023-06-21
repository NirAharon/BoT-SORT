import queue
import numpy as np

from scipy.spatial.distance import cdist
import faiss

from tracker.bot_sort import BoTSORT


class MultiCameraTracking:
    def __init__(self, args, frame_rate=30,time_window=50, global_match_thresh=0.35):

        self.time = 0
        self.last_global_id = 0
        self.global_ids_queue = queue.Queue()
        assert time_window >= 1
        self.time_window = time_window  # should be greater than time window in scts
        assert 0 <= global_match_thresh <= 1
        self.global_match_thresh = global_match_thresh
        num_sources = len(args.path)
        self.all_tracks = {}
        self.cam_id_list = []
        self.all_features = []
        self.all_track_ids = []
        self.indexes = []
        self.trackers = []
        self.num = 0
        
        
        d = 8192 
        
        for i in range(num_sources): 
            index = faiss.IndexFlatL2(d)
            self.indexes.append(index) 
        print(self.indexes)

        for i in range(num_sources):
            self.trackers.append(BoTSORT(args, frame_rate=args.fps))
        print(self.trackers)

        
    def process(self, output_results, img, cam_id):

        new_tracks = self.trackers[cam_id].update(output_results, img)
        merged = False
        for track in new_tracks: 
            index = self.indexes[cam_id]
            if track.curr_feat is not None:

                self.all_features.append(track.curr_feat)

                # self.all_track_ids.append(track.track_id)
                # self.cam_id_list.append(cam_id)

                index.add(track.curr_feat.reshape(1,-1))
                if cam_id in self.all_tracks:
                    self.all_tracks[cam_id].append(track.track_id)
                else:
                   self.all_tracks[cam_id] = [track.track_id]

            all_features = np.array(self.all_features)

            #print(self.all_tracks)

            for i in range(self.num,len(all_features)):
                self.num += 1
                query_feature = all_features[i].reshape(1, -1)
                if cam_id != 0:
                    best_distance = 1
                    for j in range(1, cam_id+1):
                        D, I = self.indexes[j - 1].search(query_feature, 1)
                        distance = D[0][0]
                        if distance < best_distance:
                            best_distance = distance
                            nearest_index = I[0][0]
                            print(nearest_index)
                            nearest_track_id = self.all_tracks[j - 1][nearest_index] 
                        else:
                            continue

                    
                    if best_distance < 0.1:
                        print("merging {} with {}".format(track.track_id, nearest_track_id))
                        merged_track = 
                        merged = True
                        #track.track_id = nearest_track_id
                else:
                    continue
        return new_tracks

class Merge:
    def __init__(self, track_id, tlwh, score):
        self.track_id = track_id
        self.tlwh = tlwh
        self.score = score
        # Other attributes...










# features = sct.get_features_keep()
# try:
#     if features.shape[0] > 0:
#         all_features = np.concatenate((all_features, features), axis=0)
# except:
#     if len(features) > 0:
#         features = np.array(features)  # Convert features to a NumPy array
#         all_features = np.concatenate((all_features, features), axis=0)

# print(all_features.shape)
# self.detections += sct.get_detections()
# print(len(self.detections))
# for i in all_tracks:
#     print(i.track_id)