from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime


class RoomDescription(BaseModel):
    """房型描述数据模型"""
    room_id: str = Field(..., description="房型唯一标识")
    description: str = Field(..., description="房型描述文本")

    @validator('description')
    def validate_description(cls, v):
        if not v or not v.strip():
            raise ValueError("房型描述不能为空")
        return v.strip()


class GroupRequest(BaseModel):
    """分组请求模型"""
    rooms: List[RoomDescription] = Field(..., description="房型描述列表")
    n_clusters: Optional[int] = Field(
        None, description="聚类数量，为None时自动计算", ge=2)
    min_cluster_size: int = Field(default=2, description="最小聚类大小", ge=2)

    @validator('rooms')
    def validate_rooms(cls, v):
        if len(v) < 2:
            raise ValueError("至少需要2个房型进行分组")
        return v


class GroupResult(BaseModel):
    """单个房型分组结果"""
    room_id: str = Field(..., description="房型ID")
    description: str = Field(..., description="原始房型描述")
    group_id: int = Field(..., description="分组ID")
    similarity_score: float = Field(...,
                                    description="与组内其他房型的平均相似度", ge=0, le=1)


class GroupResponse(BaseModel):
    """分组响应模型"""
    groups: List[GroupResult] = Field(..., description="分组结果列表")
    n_clusters: int = Field(..., description="实际聚类数量")
    group_statistics: Dict = Field(..., description="每个组的统计信息")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="分组完成时间")
