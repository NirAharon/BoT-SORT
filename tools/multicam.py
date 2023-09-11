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

        num_sources = len(args.path)
        # #self.all_tracks = {}
        #self.cam_id_list = []
        self.all_features = []
        #self.all_track_ids = []
        #self.indexes = []
        self.trackers = []
        #self.num = 0
        self.frame_id = 0
        self.person_id = 0
        

        for i in range(num_sources):
            self.trackers.append(BoTSORT(args, frame_rate=args.fps))
        print("Trackers:",self.trackers)

        
        #creating database and table

        self.conn = psycopg.connect(dbname='testdb', autocommit=True, port=5440)
        self.conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        register_vector(self.conn)
        self.conn.execute('DROP TABLE IF EXISTS detections')
        self.conn.execute('CREATE TABLE detections (id integer PRIMARY KEY, cam_id integer, activated BOOLEAN, x integer, y integer, width integer, height integer, person_id integer, person_name varchar, embedding vector(1024))')
        self.conn.execute('CREATE INDEX ON detections USING ivfflat (embedding vector_cosine_ops)')

        query = 'SELECT COUNT(*) FROM detections'
        result = self.conn.execute(query).fetchall()
        print("Result: ",result[0][0])
        if result[0][0] == 0:
            self.num = result[0][0]
        else:
            self.num = result[0][0] + 1
        
    def process(self, output_results, img, cam_id):

        #self.frame_id += 1
        active_tracks = []
        self.conn.execute('UPDATE detections SET activated = False')
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

                #Adding track to database
                query = 'INSERT INTO detections (id, cam_id, activated, x, y, width, height, person_id, person_name, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'
                self.conn.execute(query, (self.num, cam_id, False, x, y, width, height, track.track_id, 'unknown', track.curr_feat.astype(np.float32)))
                print(active_tracks, "Active tracks")
                for i in active_tracks:
                    query = 'UPDATE detections SET activated = %s WHERE person_id = %s'
                    self.conn.execute(query,(True,i))

                if self.num > 0: #if there is more than one track in the database then proceed
                    
                    query_vector = track.curr_feat.astype(np.float32)
                    #print(i)
                    #query = 'SELECT person_id, embedding, embedding::vector <-> %s::vector AS distance FROM detections WHERE cam_id != %s ORDER BY embedding::vector <-> %s::vector LIMIT 20'
                    
                    #query to select the most common person id under a certain threshold
                    most_common = 'SELECT person_id, COUNT(*) AS count \
                            FROM (SELECT person_id, embedding, embedding::vector <-> %s::vector AS distance FROM detections \
                            WHERE cam_id != %s AND embedding::vector <-> %s::vector < %s AND activated = %s ORDER BY embedding::vector <-> %s::vector LIMIT 101) \
                            AS subquery \
                            GROUP BY person_id \
                            ORDER BY count DESC \
                            LIMIT 1;'
                    most_common_result = self.conn.execute(most_common, (query_vector, cam_id, query_vector, 0.15, False, query_vector)).fetchall()
                    print("most common", most_common_result)
                    #track_ids = [row[0] for row in result if row[2] <= 0.01]


                    if len(most_common_result) == 0: #checking if there are results for the most_common query
                        cam_count = 'SELECT DISTINCT cam_id from detections'
                        cam_count_result = self.conn.execute(cam_count).fetchall()
                        if len(cam_count_result) > 1: #checking if there are more than one camera sources
                            same_common = 'SELECT person_id, COUNT(*) AS count \
                                    FROM (SELECT person_id, embedding, embedding::vector <-> %s::vector AS distance FROM detections \
                                    WHERE embedding::vector <-> %s::vector < %s AND activated = %s AND id != %s ORDER BY embedding::vector <-> %s::vector LIMIT 101) \
                                    AS subquery \
                                    GROUP BY person_id \
                                    ORDER BY count DESC \
                                    LIMIT 1;'
                            same_common_result = self.conn.execute(same_common, (query_vector, query_vector, 0.95, False, self.num, query_vector)).fetchall()
                            print("same_common",same_common_result)
                            if len(same_common_result) == 0:
                                print("no common")
                                max_personid = 'SELECT max(person_id) FROM detections WHERE id != %s'
                                max_personid_result = self.conn.execute(max_personid,(self.num,)).fetchall()
                                self.person_id = max_personid_result[0][0] + 1 
                                #increase the person id by one because this means that there is no nearest neighbor for the vector, hence a new person
                                
                                update = 'UPDATE detections SET person_id = %s WHERE id = %s' #update the person id for that detection
                                self.conn.execute(update,(self.person_id, self.num))
                                return_tracks.append(Merge(self.person_id, track.tlwh, track.score, 'unknown', cam_id))
                                active_tracks.append(self.person_id)
                            else:
                                update = 'UPDATE detections SET person_id = %s, person_name = %s WHERE id = %s'
                                self.person_id = same_common_result[0][0]
                                name_query = 'SELECT person_name FROM detections WHERE person_id = %s'
                                name_result = self.conn.execute(name_query,(self.person_id,)).fetchone()
                                self.conn.execute(update,(self.person_id, name_result[0], self.num))
                                return_tracks.append(Merge(self.person_id, track.tlwh, track.score, name_result[0], cam_id))
                                active_tracks.append(self.person_id)
                        else:
                            return_tracks.append(Merge(track.track_id,track.tlwh,track.score,'unknown', cam_id))

                    else: #when there is a result for most_common query, update the current detection with that person id
                        update = 'UPDATE detections SET person_id = %s, person_name = %s WHERE id = %s'
                        self.person_id = most_common_result[0][0]
                        name_query = 'SELECT person_name FROM detections WHERE person_id = %s'
                        name_result = self.conn.execute(name_query,(self.person_id,)).fetchone()
                        self.conn.execute(update,(self.person_id, name_result[0], self.num)) 
                        return_tracks.append(Merge(self.person_id, track.tlwh, track.score, name_result[0],cam_id))
                        active_tracks.append(self.person_id)
                    #print("frame id", self.frame_id/2, "New track id", self.person_id)
                else:
                    return_tracks.append(Merge(track.track_id,track.tlwh,track.score,'unknown',cam_id))
                self.num += 1

        return return_tracks

class Merge():
    def __init__(self, track_id, tlwh, score, name, cam_id):
        self.track_id = track_id
        self.tlwh = tlwh
        self.score = score
        self.name = name
        self.cam_id = cam_id
        # Add one more argument for cam id
























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
