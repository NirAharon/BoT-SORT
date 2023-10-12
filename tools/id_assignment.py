
import numpy as np

class TrackMerger:
    def __init__(self):
        self.global_track_id = 0
        self.track_mapping = {}
        self.merged_tracks = []
        self.number= 0

    def merge_tracks(self, tracks1, tracks2):
        for id1 in tracks1:
           # print(id1)
            self.update_track_mapping(id1, id1)
            #print(self.track_mapping)
            for id2 in tracks2:
                #print(id2)
                if self.is_similar(id1, id2):
                    #print("similar")
                    global_id = self.get_global_track_id(id1)
                    #print(global_id)
                    self.mark_as_merged(id2)
                    #print(self.merged_tracks)
                    self.update_track_mapping(id2, global_id)
                    #print(self.track_mapping)
                else:
                    #print("not similar")
                    if self.is_merged_track(id2):
                        #print("breaking")
                        continue
                    else:
                        global_id = self.get_next_global_track_id()
                        #print(global_id)
                        self.update_track_mapping(id2, global_id)
                        #print(self.track_mapping)
                if id2 not in self.merged_tracks:
                    #print("deleting")
                    #print(id2)
                    del self.track_mapping[id2]
                   # print(self.track_mapping)
        #print(self.track_mapping)
        for id2 in tracks2:
            if id2 not in self.track_mapping.keys():
                global_id = self.get_next_global_track_id()
                self.update_track_mapping(id2, global_id)
        print(self.track_mapping)


    def is_similar(self, id1, id2):
        if id1 - id2 == -4:
            return True

    def get_global_track_id(self, local_id):
            return self.track_mapping.get(local_id, local_id)

    def get_next_global_track_id(self):
            #print(self.track_mapping)
            max_global_id = max(self.track_mapping.values(), default=0)
            #print(max_global_id)
            self.global_track_id = min(self.global_track_id, max_global_id) + 1
            return self.global_track_id


    def update_track_mapping(self, local_id, global_id):
            self.track_mapping[local_id] = global_id

    def is_merged_track(self, local_id):
            if local_id in self.merged_tracks:
                return True

    def mark_as_merged(self, local_id):
            self.merged_tracks.append(local_id)


camera1_tracks = [0, 1, 2, 3]
camera2_tracks = [9, 5, 6, 8]

# Create an instance of TrackMerger
merger = TrackMerger()

# Merge the tracks
merger.merge_tracks(camera1_tracks, camera2_tracks)

# Get the global track IDs
for local_id in camera2_tracks:
    global_id = merger.get_global_track_id(local_id)
    print(f"Local ID: {local_id}, Global ID: {global_id}")
