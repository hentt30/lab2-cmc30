from __future__ import annotations
import math
from threading import Thread
import uuid
import xml.etree.ElementTree as ET
from xml.dom.minidom import parse, Node
import numpy as np

faces_vector = []
all_colors = ['r', 'g', 'b']


class Face:
    def __init__(self, p1: tuple, p2: tuple, p3: tuple, c1: tuple, c2: tuple, c3: tuple):
        self.id = uuid.uuid4()
        self.x1 = p1[0]
        self.y1 = p1[1]
        self.z1 = p1[2]
        self.x2 = p2[0]
        self.y2 = p2[1]
        self.z2 = p2[2]
        self.x3 = p3[0]
        self.y3 = p3[1]
        self.z3 = p3[2]

        self.r1 = c1[0]
        self.g1 = c1[1]
        self.b1 = c1[2]
        self.a1 = c1[3]
        self.r2 = c2[0]
        self.g2 = c2[1]
        self.b2 = c2[2]
        self.a2 = c2[3]
        self.r3 = c3[0]
        self.g3 = c3[1]
        self.b3 = c3[2]
        self.a3 = c3[3]

        self.source = 0

        pr = (self.r1 + self.r2 + self.r3) / 3
        pg = (self.g1 + self.g2 + self.g3) / 3
        pb = (self.b1 + self.b2 + self.b3) / 3
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

    def get_centroid(self) -> tuple:
        """
        Return the centroid of the triangle
        """
        return ((self.x1 + self.x2 + self.x3) / 3,
                (self.y1 + self.y2 + self.y3) / 3,
                (self.z1 + self.z2 + self.z3) / 3)

    def get_area(self) -> float:
        """
        Return the area of the triangle => S = |AxB|*(1/2)
        """
        a = (self.x2 - self.x1, self.y2 - self.y1, self.z2 - self.z1)
        b = (self.x3 - self.x1, self.y3 - self.y1, self.z3 - self.z1)
        s = 0.5 * math.sqrt((a[1] * b[2] - a[2] * b[1]) ** 2 +
                            (a[0] * b[2] - a[2] * b[0]) ** 2 +
                            (a[0] * b[1] - a[1] * b[0]) ** 2)
        return s

    def get_normal(self) -> tuple:
        """
        Get the normal of the triangle
        """
        a = (self.x2 - self.x1, self.y2 - self.y1, self.z2 - self.z1)
        b = (self.x3 - self.x1, self.y3 - self.y1, self.z3 - self.z1)
        module = self.get_area() * 2
        if module == 0.0:
            print(self.p1(), self.p2(), self.p3())
        norm = ((a[1] * b[2] - a[2] * b[1]) / module,
                -(a[0] * b[2] - a[2] * b[0]) / module,
                (a[0] * b[1] - a[1] * b[0]) / module)
        return norm

    def unit_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def determinant_3x3(self, m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                m[1][0] * (m[0][1] * m[2][2] - m[0][2] * m[2][1]) +
                m[2][0] * (m[0][1] * m[1][2] - m[0][2] * m[1][1]))

    def subtract(self, a, b):
        return (a[0] - b[0],
                a[1] - b[1],
                a[2] - b[2])

    def add(self, a, b):
        return (a[0] + b[0],
                a[1] + b[1],
                a[2] + b[2])

    def multiply(self, k, b):
        return (k * b[0], k * b[1], k * b[2])

    def dot_product(self, a, b):
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    def cross_product(self, a, b):
        return ((a[1]*b[2] - a[2]*b[1]), -(a[0]*b[2]-a[2]*b[0]), (a[0]*b[1] - a[1]*b[0]))

    def tetrahedron_calc_volume(self, a, b, c, d):
        return (self.determinant_3x3((
            self.subtract(a, b),
            self.subtract(b, c),
            self.subtract(c, d),
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

        if abs(dot_p_normal_vij) > epsilon:
            w = self.subtract(p1, plane_point)
            fac = -self.dot_product(w, plane_normal) / dot_p_normal_vij
            u = self.multiply(fac, vij)
            return self.add(p1, u)

        return None

    def same_side(self, p1, p2, a, b):
        cp1 = self.cross_product(self.subtract(b, a), self.subtract(p1, a))
        cp2 = self.cross_product(self.subtract(b, a), self.subtract(p2, a))
        if self.dot_product(cp1, cp2) >= 0:
            return True
        else:
            return False

    def point_in_triangle(self, p, a, b, c) -> bool:
        if self.same_side(p, a, b, c) and self.same_side(p, b, a, c) and self.same_side(p, c, a, b):
            return True
        else:
            return False

    def p1(self) -> tuple():
        return (self.x1, self.y1, self.z1)

    def p2(self) -> tuple():
        return (self.x2, self.y2, self.z2)

    def p3(self) -> tuple():
        return (self.x3, self.y3, self.z3)

    def get_factor_form(self, face: Face) -> float:
        """
        return the factor form used to solve the linear system
        """
        vij = (self.centroid[0] - face.centroid[0], self.centroid[1] -
               face.centroid[1], self.centroid[2] - face.centroid[2])
        if self.id == face.id or (180.0/math.pi) * self.angle_between(vij, face.normal) < 90:
            return 0

        # check if the triangles aren't blocked
        for other_face in faces_vector:
            if other_face.id == face.id or other_face.id == self.id:
                continue
            v1 = self.tetrahedron_calc_volume((other_face.x1, other_face.y1, other_face.z1), (
                other_face.x2, other_face.y2, other_face.z2), (other_face.x3, other_face.y3, other_face.z3), self.centroid)
            v2 = self.tetrahedron_calc_volume((other_face.x1, other_face.y1, other_face.z1), (
                other_face.x2, other_face.y2, other_face.z2), (other_face.x3, other_face.y3, other_face.z3), face.centroid)
            if np.sign(v1) == np.sign(v2):
                continue
            intersect = self.intersect_segment_plane(
                self.centroid, face.centroid, (other_face.x1, other_face.y1, other_face.z1), other_face.normal)
            if intersect is not None:
                result = self.point_in_triangle(
                    intersect, other_face.p1(), other_face.p2(), other_face.p3())
                if result:
                    return 0

        # Return the view factor
        Aj = face.area
        cos_theta_i = np.cos(self.angle_between(vij, self.centroid))
        cos_theta_j = np.cos(self.angle_between(vij, face.centroid))
        r = np.linalg.norm(vij)

        return (Aj * cos_theta_i * cos_theta_j) / (math.pi * (r ** 2) + Aj)


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
            F = faces[i].get_factor_form(faces[j])
            a[i][j] -= faces[i].reflect[color] * F

    x = np.linalg.solve(a, b)

    for i in range(len(x)):
        faces[i].ilumination[color] = x[i]


if __name__ == "__main__":
    # Parse the system
    document = parse('exemplocena.dae')
    set_id_attribute(document)
    root = document.documentElement
    # tree = ET.parse('exemplocena.dae')
    # root = tree.getroot()
    # elm = root.find('library_geometries')
    # objects = {}
    # for geometry in elm:
    #     print(geometry.attrib)
    #     objects[geometry.attrib['id']] = {}
    #     elements = geometry[0].findall('source')
    #     pos = list(filter(
    #         lambda x: x.attrib['id'] == geometry.attrib['id'] + '-positions',
    #         elements))[0]
    #     print(pos.find('float_array').text)

    #     triangles = geometry[0].find('triangles')
    #     p = triangles.find('p')
    #     indexes = list(map(int, p.text.split()))

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

    print(objects.keys())
    print(objects['Cube'].keys())

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
            p1 = vertices[j], vertices[j + 1], vertices[j + 2]
            k = 3 * indexes[i + 4]
            p2 = vertices[k], vertices[k + 1], vertices[k + 2]
            m = 3 * indexes[i + 8]
            p3 = vertices[m], vertices[m + 1], vertices[m + 2]

            j = 4 * indexes[i + 3]
            c1 = colors[j], colors[j + 1], colors[j + 2], colors[j + 3]
            k = 4 * indexes[i + 7]
            c2 = colors[k], colors[k + 1], colors[k + 2], colors[k + 3]
            m = 4 * indexes[i + 11]
            c3 = colors[m], colors[m + 1], colors[m + 2], colors[m + 3]

            face = Face(p1, p2, p3, c1, c2, c3)
            reverse[id][p1.__str__()] = (face, 1, i)
            reverse[id][p2.__str__()] = (face, 2, i)
            reverse[id][p3.__str__()] = (face, 3, i)
            object['faces'].append(face)
            faces_vector.append(face)

            i += 12

    # Solve the linear system (ax = b) for each color
    # threads = []
    for c in all_colors:
        set_final_iluminations(faces_vector, c)
        # t = Thread(target=set_final_iluminations, args=(faces_vector, c))
        # t.start()
        # threads.append(t)
    # for t in threads:
        # t.join()

    # Updates the vertices colors of the faces
    for obj in objects:
        for point, faces in reverse[obj].items():
            n = len(faces)
            rm = gm = bm = 0
            for face in faces:
                rm += face[0]['r' + str(face[1])] / n
                gm += face[0]['g' + str(face[1])] / n
                bm += face[0]['b' + str(face[1])] / n
            for face in faces:
                j = indexes[face[2] + 3 + 4 * (face[1] - 1)]
                objects[obj]['colors'][j] = rm
                objects[obj]['colors'][j + 1] = gm
                objects[obj]['colors'][j + 2] = bm

    # Update the files
