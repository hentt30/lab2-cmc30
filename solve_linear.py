from __future__ import annotations
import math
import uuid
import xml.etree.ElementTree as ET
from xml.dom.minidom import parse, Node

class Face:
    def __init__(self,id:uuid,p1:tuple,p2:tuple,p3:tuple,c1:tuple,c2:tuple,c3:tuple):
        self.id = id
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
        self.pr = (self.r1 + self.r2 + self.r3)/3
        self.pg = (self.g1 + self.g2 + self.g3)/3
        self.pb = (self.b1 + self.b2 + self.b3)/3
        self.centroid = self.get_centroid()
        self.area = self.get_area()
        self.norm = self.get_norm()
    
    def get_centroid(self)->tuple:
        """
        Return the centroid of the triangle
        """
        return ((self.x1 + self.x2 + self.x3)/3,(self.y1 + self.y2 + self.y3)/3,(self.z1 + self.z2 + self.z3)/3)
    
    def get_area(self)->float:
        """
        Return the area of the triangle => S = |AxB|*(1/2)
        """
        a = (self.x2 - self.x1,self.y2 - self.y1,self.z2 - self.z1)
        b = (self.x3 - self.x1,self.y3 - self.y1,self.z3 - self.z1)
        s = 0.5*math.sqrt((a[1]*b[2] - a[2]*b[1])**2 + (a[0]*b[2]-a[2]*b[0])**2 + (a[0]*b[1] - a[1]*b[0])**2)
        return s
    
    def get_normal(self)->tuple:
        """
        Get the normal of the triangle
        """
        a = (self.x2 - self.x1,self.y2 - self.y1,self.z2 - self.z1)
        b = (self.x3 - self.x1,self.y3 - self.y1,self.z3 - self.z1)
        module = self.get_area()*2
        norm = ((a[1]*b[2] - a[2]*b[1])/module , -(a[0]*b[2]-a[2]*b[0])/module , (a[0]*b[1] - a[1]*b[0])/module)
        return norm
    
    def get_factor_form(self,face:Face)->float:
        """
        return the factor form used to solve the linear system   
        """
        if self.id == face.id:
            a = 4

        pass


def set_id_attribute(parent, attribute_name="id"):
    """
    Set id attributes on xml files
    """
    if parent.nodeType == Node.ELEMENT_NODE:
        if parent.hasAttribute(attribute_name):
             parent.setIdAttribute(attribute_name)
    for child in parent.childNodes:
        set_id_attribute(child, attribute_name)

## Parse the system
document = parse('exemplocena.dae')
set_id_attribute(document)
root = document.documentElement

objects = {}

for geometry in root.getElementsByTagName("float_array"):
    print(geometry.getAttribute("id"))
    id = geometry.getAttribute("id")
    splitted_id = id.split('-')
    if splitted_id[0] not in objects:
        objects[splitted_id[0]] = {}
    objects[splitted_id[0]][splitted_id[2]] = list(map(float,geometry.firstChild.data.split()))

## Create the faces objects

## Solve the linear system

## Update the files
