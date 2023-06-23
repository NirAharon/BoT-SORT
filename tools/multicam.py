import queue
import numpy as np
from pgvector.psycopg import register_vector
import psycopg


from scipy.spatial.distance import cdist
import faiss

from tracker.bot_sort import BoTSORT
from collections import Counter


class MultiCameraTracking:
    def __init__(self, args, frame_rate=30,time_window=50, global_match_thresh=0.35):

        # self.time = 0
        # self.last_global_id = 0
        # self.global_ids_queue = queue.Queue()
        # assert time_window >= 1
        # self.time_window = time_window  # should be greater than time window in scts
        # assert 0 <= global_match_thresh <= 1
        # self.global_match_thresh = global_match_thresh
        num_sources = len(args.path)
        # #self.all_tracks = {}
        #self.cam_id_list = []
        self.all_features = []
        #self.all_track_ids = []
        #self.indexes = []
        self.trackers = []
        self.num = 0
        self.frame_id = 0
        self.person_id = 0
        
        # d = 8192 
        
        # for i in range(num_sources): 
        #     index = faiss.IndexFlatL2(d)
        #     self.indexes.append(index) 
        # print(self.indexes)

        for i in range(num_sources):
            self.trackers.append(BoTSORT(args, frame_rate=args.fps))
        print(self.trackers)


        self.conn = psycopg.connect(dbname='testdb', autocommit=True)
        self.conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        register_vector(self.conn)
        self.conn.execute('DROP TABLE IF EXISTS detections')
        self.conn.execute('CREATE TABLE detections (id integer PRIMARY KEY, cam_id integer, track_id integer, x integer, y integer, width integer, height integer, person_id integer, embedding vector(1024))')
        self.conn.execute('CREATE INDEX ON detections USING ivfflat (embedding vector_cosine_ops)')
        
    def process(self, output_results, img, cam_id):
        merged = False
        self.frame_id += 1
        new_tracks = self.trackers[cam_id].update(output_results, img)
        return_tracks = []
        for track in new_tracks: 
            #index = self.indexes[cam_id]
            if track.curr_feat is not None:
                x = track.tlwh[0]
                y = track.tlwh[1]
                width = track.tlwh[2]
                height = track.tlwh[3]

                # query = 'SELECT COUNT(*) FROM detections'

                # result = self.conn.execute(query).fetchall()
                # if result[0][0] < 100:
                print(self.num)
                query = 'INSERT INTO detections (id, cam_id, track_id, x, y, width, height, person_id, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)'
                self.conn.execute(query, (self.num, cam_id, track.track_id, x, y, width, height, track.track_id, track.curr_feat.astype(np.float32)))

                if self.num > 0:
                    
                    query_vector = track.curr_feat.astype(np.float32)
                    #print(i)
                    #query = 'SELECT person_id, embedding, embedding::vector <-> %s::vector AS distance FROM detections WHERE cam_id != %s ORDER BY embedding::vector <-> %s::vector LIMIT 20'
                    query = 'SELECT person_id, COUNT(*) AS count \
                            FROM (SELECT person_id, embedding, embedding::vector <-> %s::vector AS distance FROM detections \
                            WHERE cam_id != %s AND embedding::vector <-> %s::vector < %s ORDER BY embedding::vector <-> %s::vector LIMIT 101) \
                            AS subquery \
                            GROUP BY person_id \
                            ORDER BY count DESC \
                            LIMIT 1;'
                    result = self.conn.execute(query, (query_vector, cam_id, query_vector, 0.01, query_vector)).fetchall()
                    print(result)
                    #track_ids = [row[0] for row in result if row[2] <= 0.01]

                    if len(result) <= 0:
                        query = 'SELECT DISTINCT cam_id from detections'
                        result = self.conn.execute(query).fetchall()
                        if len(result) > 1:
                            query = 'SELECT max(person_id) FROM detections WHERE id != %s'
                            result = self.conn.execute(query,(self.num,)).fetchall()
                            self.person_id = result[0][0] + 1

                            query = 'UPDATE detections SET person_id = %s WHERE id = %s'
                            self.conn.execute(query,(self.person_id, self.num))
                            return_tracks.append(Merge(self.person_id, track.tlwh, track.score))
                            merged = True

                    else:
                        query = 'UPDATE detections SET person_id = %s WHERE id = %s'
                        self.person_id = result[0][0]
                        self.conn.execute(query,(self.person_id,self.num))
                        return_tracks.append(Merge(self.person_id, track.tlwh, track.score))
                        merged = True
                    print("frame id", self.frame_id/2, "New track id", self.person_id)

                self.num += 1
                        
                    # # Count the occurrences of each track_id
                    #     track_id_counts = Counter(track_ids)
                    #     print(track_id_counts)
                    #     # Find the most common track_id
                    #     most_common_track_id = track_id_counts.most_common(1)[0][0]

                    #     query = 'INSERT INTO detections (cam_id, track_id, x, y, width, height, person_id, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'
                    #     self.conn.execute(query, (cam_id, track.track_id, x, y, width, height, most_common_track_id, track.curr_feat.astype(np.float32)))
                    #     new_id = most_common_track_id
                    # else:
                    #     query = 'SELECT max(person_id) FROM detections'
                    #     result = self.conn.execute(query).fetchall()
                    #     self.person_id = result[0][0] + 1
                    #     query = 'INSERT INTO detections (cam_id, track_id, x, y, width, height, person_id, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'
                    #     self.conn.execute(query, (cam_id, track.track_id, x, y, width, height, self.person_id, track.curr_feat.astype(np.float32)))
                    #     new_id = self.person_id

                # return_tracks.append(Merge(new_id, track.tlwh, track.score))
                # merged = True
                # print(return_tracks)

        if merged == True:
            return return_tracks
        else:
            return new_tracks

class Merge():
    def __init__(self, track_id, tlwh, score):
        self.track_id = track_id
        self.tlwh = tlwh
        self.score = score


























            #print(self.all_tracks)

            # for i in range(self.num,len(all_features)):
            #     self.num += 1
            #     query_feature = all_features[i].reshape(1, -1)
            #     if cam_id != 0:
            #         best_distance = 1
            #         for j in range(1, cam_id+1):
            #             D, I = self.indexes[j - 1].search(query_feature, 1)
            #             distance = D[0][0]
            #             if distance < best_distance:
            #                 best_distance = distance
            #                 nearest_index = I[0][0]
            #                 print(nearest_index)
            #                 nearest_track_id = self.all_tracks[j - 1][nearest_index] 
            #             else:
            #                 continue

                    
            #         if best_distance < 0.1:
            #             print("merging {} with {}".format(track.track_id, nearest_track_id))
            #             merged = True
            #             #track.track_id = nearest_track_id
            #     else:
            #         continue












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
