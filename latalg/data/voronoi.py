from __future__ import annotations
import random
from typing import Callable, NamedTuple
from jaxtyping import Float, Int

import torch
from torch import Tensor
    
VoronoiShape = NamedTuple('VoronoiShape', [
    ('inshape_points', Float[Tensor, 'inshape_point_n point_d']),
    ('outshape_points', Float[Tensor, 'outshape_point_n point_d']),
])

def make_voronoi_shape(voronoi_random_max: int) -> VoronoiShape:
    inshape_point_n = random.randint(1, voronoi_random_max)
    outshape_point_n = random.randint(1, voronoi_random_max)
    
    point_d = 2
    inshape_points = torch.rand(inshape_point_n, point_d)
    outshape_points = torch.rand(outshape_point_n, point_d) 
    
    return VoronoiShape(inshape_points, outshape_points)

def voronoi_target_func(shape: VoronoiShape) -> Callable:
    inshape_points = shape.inshape_points
    outshape_points = shape.outshape_points

    def target_function(x: Float[Tensor, 'b point_d']) -> Int[Tensor, 'b']:
        dist_to_inshape = torch.cdist(x.unsqueeze(0), inshape_points.unsqueeze(0))
        dist_to_outshape = torch.cdist(x.unsqueeze(0), outshape_points.unsqueeze(0))
        closest_inshape = torch.min(dist_to_inshape, dim=2).values
        closest_outshape = torch.min(dist_to_outshape, dim=2).values
        inshape = (closest_inshape < closest_outshape).squeeze(0)
        return inshape.long().unsqueeze(1)
    
    return target_function

def union_target_func(shape1: VoronoiShape, shape2: VoronoiShape) -> Callable:
    func1 = voronoi_target_func(shape1)
    func2 = voronoi_target_func(shape2)
    
    def target_function(x: Float[Tensor, 'b point_d']) -> Int[Tensor, 'b']:
        return torch.maximum(func1(x), func2(x))
    
    return target_function

def intersection_target_func(shape1: VoronoiShape, shape2: VoronoiShape) -> Callable:
    func1 = voronoi_target_func(shape1)
    func2 = voronoi_target_func(shape2)
    
    def target_function(x: Float[Tensor, 'b point_d']) -> Int[Tensor, 'b']:
        return torch.minimum(func1(x), func2(x))
    
    return target_function

    
def pad_points_to_max_len(
        points: Float[Tensor, 'point_n point_d']
    ) -> Float[Tensor, 'max_point_n point_d']:
    
    max_point_n = 10
    if points.shape[0] > max_point_n:
        raise ValueError(f'points.shape[0] = {points.shape[0]} > {max_point_n}')
    
    # Pad with -1000
    points = torch.cat([
        points,
        torch.ones(max_point_n - points.shape[0], points.shape[1]) * -1000,
    ], dim=0)
    
    return points

def remove_padding_from_points(
        points: Float[Tensor, 'max_point_n point_d']
    ) -> Float[Tensor, 'point_n point_d']:
    
    return points[points[:, 0] > -999]
