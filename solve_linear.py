from __future__ import annotations
import math
from threading import Thread
import uuid
import xml.etree.ElementTree as ET
from xml.dom.minidom import parse, Node
import numpy as np

faces_vector = []
faces_p1 = np.empty((0,3), int)
faces_p2 = np.empty((0,3), int)
faces_p3 = np.empty((0,3), int)
all_colors = ['r', 'g', 'b']

matrices = {}
debugint = []

class Face:
    
    def __init__(self, object_id, p1: np.array, p2: np.array, p3: np.array, c1: np.array, c2: np.array, c3: np.array):
        self.id = uuid.uuid4()
        self.object_id = object_id
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

        self.source = {
            'r': 0,
            'g': 0,
            'b': 0,
        }

        pr = (self.c1[0] + self.c2[0] + self.c3[0]) / 3.0
        pg = (self.c1[1] + self.c2[1] + self.c3[1]) / 3.0
        pb = (self.c1[2] + self.c2[2] + self.c3[2]) / 3.0

        self.reflect = {
            'r': pr,
            'g': pg,
            'b': pb,
        }

        self.ilumination = {
            'r': 0,
            'g': 0,
            'b': 0,
        }

        self.centroid = self.get_centroid()
        self.area = self.get_area()
        self.normal = self.get_normal()
        if self.object_id == "Cube_002":
            self.normal *= -1
        self.cache = {}

    def get_centroid(self) -> tuple:
        """
        Return the centroid of the triangle
        """
        return (self.p1 + self.p2 + self.p3)/3

    def get_area(self) -> float:
        """
        Return the area of the triangle => S = |AxB|*(1/2)
        """
        a = self.p2 - self.p1
        b = self.p3 - self.p1
        return 0.5 * np.linalg.norm(np.cross(a,b))
        

    def get_normal(self) -> tuple:
        """
        Get the normal of the triangle
        """
        a = self.p2 - self.p1
        b = self.p3 - self.p1
        return np.cross(a,b)/(np.linalg.norm(np.cross(a,b)))

    def unit_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def same_side(self, p1, p2, a, b):
        cp1 = np.cross(b-a, p1-a)
        cp2 = np.cross(b-a, p2-a)
        return np.sum(cp1*cp2, axis=1) >= 0

    def point_in_triangle(self, p, a, b, c) -> bool:
        return self.same_side(p, a, b, c) & self.same_side(p, b, a, c) & self.same_side(p, c, a, b)
    
    def get_view_factor(self, face: Face) -> float:
        """
        return the factor form used to solve the linear system
        """
        if face.id in self.cache:
            return self.cache[face.id]
        vij = face.centroid - self.centroid
        vji = self.centroid - face.centroid
        if self.id == face.id:
            return 0
        if ( self.angle_between(vij, face.normal) <= np.pi/2 or self.angle_between(vji, self.normal) <= np.pi/2):
            return 0
        def intersect_line_triangle(q1,q2,p1,p2,p3):
            def signed_tetra_volume(a,b,c,d):
                return np.sum(np.cross(b-a,c-a)*(d-a),axis=1)/6.0

            s1 = signed_tetra_volume(q1,p1,p2,p3)
            s2 = signed_tetra_volume(q2,p1,p2,p3)
            s3 = signed_tetra_volume(q1,q2,p1,p2)
            s4 = signed_tetra_volume(q1,q2,p2,p3)
            s5 = signed_tetra_volume(q1,q2,p3,p1)
            x = (np.sign(s1)!=np.sign(s2))
            y = (np.sign(s3)==np.sign(s4))
            z = (np.sign(s4)==np.sign(s5))
            w = ((np.abs(s1) > 1e-9) & (np.abs(s2) > 1e-9) & (np.abs(s3) > 1e-9) & (np.abs(s4) > 1e-9) & (np.abs(s5) > 1e-9))
            return x&y&z&w

        self_centroid = np.array([self.centroid]*len(faces_p1))
        face_centroid = np.array([face.centroid]*len(faces_p1))
        res = intersect_line_triangle(self_centroid,face_centroid,faces_p1,faces_p2,faces_p3)
        if(np.sum(res) > 0):
            if np.sum(res) not in debugint:
                debugint.append(np.sum(res))
                print(np.sum(res), self.object_id, face.object_id, self.normal, face.normal, self.centroid, face.centroid)
                v = np.array(faces_vector)[res]
                for f in v:
                    print(f.object_id, f.normal, end = ' ')
                print("")
            return 0
        # Return the view factor
        Aj = face.area
        cos_theta_i = np.cos(self.angle_between(vij, self.normal))
        cos_theta_j = -np.cos(self.angle_between(vij, face.normal))
        r = np.linalg.norm(vij)
        self.cache[face.id] = (Aj * cos_theta_i * cos_theta_j) / (math.pi * (r ** 2))
        return self.cache[face.id]


def set_id_attribute(parent, attribute_name="id"):
    """
    Set id attributes on xml files
    """
    if parent.nodeType == Node.ELEMENT_NODE:
        if parent.hasAttribute(attribute_name):
            parent.setIdAttribute(attribute_name)
    for child in parent.childNodes:
        set_id_attribute(child, attribute_name)


def set_final_iluminations(faces: list[Face], color: str) -> np.ndarray:
    size = len(faces)
    a = np.identity(size)
    b = np.zeros(size)
    for i in range(size):
        b[i] = faces[i].source[color]
        for j in range(size):
            F = faces[i].get_view_factor(faces[j])
            a[i][j] -= faces[i].reflect[color] * F
    x = np.linalg.solve(a, b)
    x = np.clip(8*x,0,1)
    for i in range(len(x)):
        faces[i].ilumination[color] = x[i]


if __name__ == "__main__":
    # Parse the system
    document = parse('exemplocena.dae')
    set_id_attribute(document)
    root = document.documentElement

    objects = {}
    reverse = {}

    for geometry in root.getElementsByTagName("geometry"):
        id = ''
        for arr in geometry.getElementsByTagName("float_array"):
            id = arr.getAttribute("id")
            splitted_id = id.split('-')
            if splitted_id[0] not in objects:
                objects[splitted_id[0]] = {}
            objects[splitted_id[0]][splitted_id[2]] = list(
                map(float, arr.firstChild.data.split()))
        for arr in geometry.getElementsByTagName("p"):
            splitted_id = id.split('-')
            if splitted_id[0] not in objects:
                objects[splitted_id[0]] = {}
            objects[splitted_id[0]]['mapping'] = list(
                map(int, arr.firstChild.data.split()))
    for node in root.getElementsByTagName("node"):
        id = node.getAttribute("id")
        for arr in node.getElementsByTagName("matrix"):
            matrices[id] = np.array(list(map(float, arr.firstChild.data.split()))).reshape((4,4))


    # Create the faces objects
    for id, object in objects.items():
        vertices = object['positions']
        colors = object['colors']
        indexes = object['mapping']

        object['faces'] = []
        reverse[id] = {}
        i = 0
        while i < len(indexes):
            j = 3 * indexes[i]
            p1 = np.matmul(matrices[id],np.array([vertices[j], vertices[j + 1], vertices[j + 2],1]))[:3]
            k = 3 * indexes[i + 4]
            p2 = np.matmul(matrices[id],np.array([vertices[k], vertices[k + 1], vertices[k + 2],1]))[:3]
            m = 3 * indexes[i + 8]
            p3 = np.matmul(matrices[id],np.array([vertices[m], vertices[m + 1], vertices[m + 2],1]))[:3]
          

            a = 4 * indexes[i + 3]
            c1 = np.array([colors[a], colors[a + 1], colors[a + 2], colors[a + 3]])
            b = 4 * indexes[i + 7]
            c2 = np.array([colors[b], colors[b + 1], colors[b + 2], colors[b + 3]])
            c = 4 * indexes[i + 11]
            c3 = np.array([colors[c], colors[c + 1], colors[c + 2], colors[c + 3]])
            source = {'r': 1, 'g': 1, 'b': 1} if id == 'Cube_001' else {'r': 0, 'g': 0, 'b': 0}
            face = Face(id, p1, p2, p3, c1, c2, c3)
            face.source = source
            if j not in reverse[id]:
                reverse[id][j] = []
            if k not in reverse[id]:
                reverse[id][k] = []
            if m not in reverse[id]:
                reverse[id][m] = []
            reverse[id][j].append((face, 1, i))
            reverse[id][k].append((face, 2, i))
            reverse[id][m].append((face, 3, i))

            object['faces'].append(face)
            faces_vector.append(face)
            faces_p1=np.append(faces_p1,np.array([p1.copy()]),axis=0)
            faces_p2=np.append(faces_p2,np.array([p2.copy()]),axis=0)
            faces_p3=np.append(faces_p3,np.array([p3.copy()]),axis=0)

            i += 12
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    debug = False
    if debug:
        triangles =  [(face.p1,face.p2,face.p3) for face in faces_vector]
        f = 1
        u = [face.normal[0]/f + face.centroid[0] for face in faces_vector]
        v = [face.normal[1]/f + face.centroid[1] for face in faces_vector]
        w = [face.normal[2]/f + face.centroid[2] for face in faces_vector]
        x = [face.centroid[0] for face in faces_vector]
        y = [face.centroid[1] for face in faces_vector]
        z = [face.centroid[2] for face in faces_vector]

        ax = plt.gca(projection="3d")

        #ax.add_collection(Poly3DCollection(triangles))
        for i in range(len(x)):
            ax.plot([x[i],u[i]],[y[i],v[i]],[z[i],w[i]])
        #ax.quiver(x,y,z,u,v,w)
        ax.scatter(x, y, z, marker='o',color='r')
        ax.set_xlim([-5,5])
        ax.set_ylim([-5,5])
        ax.set_zlim([-5,5])

        plt.show()

    for c in all_colors:
        set_final_iluminations(faces_vector, c)

    # Updates the vertices colors of the faces
    for obj in objects:
        nindexes = objects[obj]['mapping']
        for point, faces in reverse[obj].items():
            n = len(faces)
            rm = gm = bm = 0
            for face in faces:
                rm += getattr(face[0],'ilumination')['r'] / n
                gm += getattr(face[0],'ilumination')['g'] / n
                bm += getattr(face[0],'ilumination')['b'] / n
            for face in faces:
                j = 4*nindexes[face[2] + 3 + 4 * (face[1] - 1)]
                objects[obj]['colors'][j] = rm
                objects[obj]['colors'][j + 1] = gm
                objects[obj]['colors'][j + 2] = bm



    ## Update the files
    for geometry in root.getElementsByTagName("geometry"):
        id = ''
        for arr in geometry.getElementsByTagName("float_array"):
            id = arr.getAttribute("id")
            splitted_id = id.split('-')
            if splitted_id[2] != 'colors':
                continue
            arr.firstChild.data = ' '.join([str(x) for x in objects[splitted_id[0]][splitted_id[2]]])

    with open('solver.dae','w') as file:
        file.write(root.toxml())