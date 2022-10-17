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

matrices = {"Cube_002":np.array([[-1.567307 ,0 ,0, -0.01589326], [0 ,-1.567307, 0, 0.372328], [0, 0, -0.1410459, 0.651979] ,[0, 0, 0, 1]]),
"Cube_001":np.array([[0.3195248, 0, 0, -0.4411952], [0, 0.3195248, 0, -2.296521], [0, 0, 0.3195248, 2.502618], [0, 0, 0, 1]]),
"Cylinder":np.array([[1, 0, 0, 0.1575703], [0, 1, 0, 0.2751122], [0, 0, 1, -0.7786573], [0, 0, 0, 1]]),
"Cube":np.array([[3 ,0 ,0 ,0] ,[0 ,3 ,0, -0.02454412] ,[0 ,0 ,3 ,1] ,[0 ,0 ,0 ,1]]),}

class Face:
    
    def __init__(self, p1: np.array, p2: np.array, p3: np.array, c1: np.array, c2: np.array, c3: np.array):
        self.id = uuid.uuid4()
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

        self.source = 0

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

    def determinant_3x3(self, m):
        return np.linalg.det(m)

    def subtract(self, a, b):
        return a-b

    def add(self, a, b):
        return  a + b

    def multiply(self, k, b):
        return k*b

    def dot_product(self, a, b):
        return np.dot(a, b)

    def cross_product(self, a, b):
        return np.cross(a, b)

    def tetrahedron_calc_volume(self, a, b, c, d):
        return (self.determinant_3x3(np.array([
            a-b,
            b-c,
            c-d,]
        )) / 6.0)

    def intersect_segment_plane(self, p1, p2, plane_point, plane_normal, epsilon=1e-6):
        """
        p1: first point of the segment
        p2: other point of the segment
        plane_point: onepoint of the plane
        plane_normal: the norm of the plane
        """
        vij = self.subtract(p1, p2)
        dot_p_normal_vij = self.dot_product(vij, plane_normal)

        condition = ~(abs(dot_p_normal_vij) > epsilon)
        w = self.subtract(p1, plane_point)
        fac = -self.dot_product(w, plane_normal) / (dot_p_normal_vij + int(condition))
        u = self.multiply(fac, vij)
        res = self.add(p1, u)

        return np.where(condition,None,res)

    def same_side(self, p1, p2, a, b):
        cp1 = self.cross_product(self.subtract(b, a), self.subtract(p1, a))
        cp2 = self.cross_product(self.subtract(b, a), self.subtract(p2, a))
        return self.dot_product(cp1, cp2) >= 0

    def point_in_triangle(self, p, a, b, c) -> bool:
        return self.same_side(p, a, b, c) and self.same_side(p, b, a, c) and self.same_side(p, c, a, b)
    
    

    def get_view_factor(self, face: Face) -> float:
        """
        return the factor form used to solve the linear system
        """
        if face.id in self.cache:
            return self.cache[face.id]
        vij = face.centroid - self.centroid
        if self.id == face.id:
            return 0
        if ( self.angle_between(vij, face.normal) <= np.pi/2 or self.angle_between(vij, self.normal) > np.pi/2):
            return 0

        def intersect_line_triangle(q1,q2,p1,p2,p3):
            def signed_tetra_volume(a,b,c,d):
                return np.sign(np.sum(np.cross(b-a,c-a)*(d-a),axis=1)/6.0)

            s1 = signed_tetra_volume(q1,p1,p2,p3)
            s2 = signed_tetra_volume(q2,p1,p2,p3)
            s3 = signed_tetra_volume(q1,q2,p1,p2)
            s4 = signed_tetra_volume(q1,q2,p2,p3)
            s5 = signed_tetra_volume(q1,q2,p3,p1)
            x = (s1!=s2)
            y = (s3==s4)
            z = (s4==s5)
            return x&y&z

        self_centroid = np.array([self.centroid]*len(faces_p1))
        face_centroid = np.array([face.centroid]*len(faces_p1))
        res = intersect_line_triangle(self_centroid,face_centroid,faces_p1,faces_p2,faces_p3)

        if(np.sum(res) > 2):
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
        b[i] = faces[i].source
        for j in range(size):
            F = faces[i].get_view_factor(faces[j])
            a[i][j] -= faces[i].reflect[color] * F
    x = np.linalg.solve(a, b)/300
    x = np.clip(x,0,1)
    for i in range(len(x)):
        #faces[i].ilumination[color] = x[i] #/ (max(x)-min(x)) - min(x) / (max(x) - min(x))
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
            p1 = np.matmul(np.array([vertices[j], vertices[j + 1], vertices[j + 2],1]),matrices[id])[:3]
            k = 3 * indexes[i + 4]
            p2 = np.matmul(np.array([vertices[k], vertices[k + 1], vertices[k + 2],1]),matrices[id])[:3]
            m = 3 * indexes[i + 8]
            p3 = np.matmul(np.array([vertices[m], vertices[m + 1], vertices[m + 2],1]),matrices[id])[:3]


            j = 4 * indexes[i + 3]
            c1 = np.array([colors[j], colors[j + 1], colors[j + 2], colors[j + 3]])
            k = 4 * indexes[i + 7]
            c2 = np.array([colors[k], colors[k + 1], colors[k + 2], colors[k + 3]])
            m = 4 * indexes[i + 11]
            c3 = np.array([colors[m], colors[m + 1], colors[m + 2], colors[m + 3]])
            source = 200 if id == 'Cylinder' else 0
            face = Face(p1, p2, p3, c1, c2, c3)
            face.source = source
            if p1.__str__() not in reverse[id]:
                reverse[id][p1.__str__()] = []
            if p2.__str__() not in reverse[id]:
                reverse[id][p2.__str__()] = []
            if p3.__str__() not in reverse[id]:
                reverse[id][p3.__str__()] = []
            reverse[id][p1.__str__()].append((face, 1, i))
            reverse[id][p2.__str__()].append((face, 2, i))
            reverse[id][p3.__str__()].append((face, 3, i))
            if(i == 2160):
                print("")
            if i > 4300:
                print("")
            object['faces'].append(face)
            faces_vector.append(face)
            faces_p1=np.append(faces_p1,np.array([p1]),axis=0)
            faces_p2=np.append(faces_p2,np.array([p2]),axis=0)
            faces_p3=np.append(faces_p3,np.array([p3]),axis=0)

            i += 12

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