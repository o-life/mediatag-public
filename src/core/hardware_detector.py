"""硬件检测和智能并发配置"""
import os
import subprocess
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class HardwareDetector:
    """硬件检测器 - 基于 Apple M 芯片 + SSD 优化"""
    
    # 最低配置要求
    MIN_REQUIREMENTS = {
        "cpu_count": 8,  # M1/M2 基础版
        "memory_gb": 16,
    }
    
    # VLM 模型内存需求（MiniCPM-V 约 8-10GB）
    VLM_MEMORY_PER_INSTANCE = 12  # GB（包含缓冲）
    SYSTEM_MEMORY_RESERVE = 6  # GB（系统预留）
    
    @staticmethod
    def detect() -> Dict[str, Any]:
        """检测硬件配置"""
        # CPU 核心数
        cpu_count = os.cpu_count() or 8
        
        # 内存大小 (GB)
        try:
            mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            mem_gb = mem_bytes / (1024**3)
        except:
            mem_gb = 16
        
        # GPU 信息 (Apple Silicon)
        gpu_cores = 0
        chip_name = "Unknown"
        
        try:
            result = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType'],
                capture_output=True, text=True, timeout=5
            )
            output = result.stdout
            
            # 查找芯片名称
            for line in output.split('\n'):
                if 'Chipset Model' in line:
                    chip_name = line.split(':')[-1].strip()
                    break
            
            # 查找 GPU 核心数
            for line in output.split('\n'):
                if 'Total Number of Cores' in line:
                    gpu_cores = int(''.join(filter(str.isdigit, line)))
                    break
        except Exception as e:
            logger.warning(f"GPU 检测失败: {e}")
        
        # 检测是否为 Apple Silicon
        is_apple_silicon = chip_name.lower().startswith('apple')
        
        return {
            "cpu_count": cpu_count,
            "memory_gb": round(mem_gb, 1),
            "gpu_cores": gpu_cores,
            "chip_name": chip_name,
            "is_apple_silicon": is_apple_silicon,
            "disk_type": "SSD"  # 固定为 SSD
        }
    
    @staticmethod
    def check_minimum_requirements(hardware: Dict[str, Any]) -> Dict[str, Any]:
        """检查是否满足最低要求"""
        min_req = HardwareDetector.MIN_REQUIREMENTS
        
        checks = {
            "cpu": hardware["cpu_count"] >= min_req["cpu_count"],
            "memory": hardware["memory_gb"] >= min_req["memory_gb"],
        }
        
        all_pass = all(checks.values())
        
        issues = []
        if not checks["cpu"]:
            issues.append(f"CPU 核心数不足（需要 {min_req['cpu_count']} 核，当前 {hardware['cpu_count']} 核）")
        if not checks["memory"]:
            issues.append(f"内存不足（需要 {min_req['memory_gb']}GB，当前 {hardware['memory_gb']}GB）")
        
        return {
            "meets_minimum": all_pass,
            "checks": checks,
            "issues": issues
        }
    
    @staticmethod
    def classify_device(hardware: Dict[str, Any]) -> str:
        """设备分级：basic/pro/max/ultra"""
        cpu = hardware["cpu_count"]
        mem = hardware["memory_gb"]
        
        # 基于 M 芯片系列分级
        if cpu >= 16 and mem >= 64:
            return "ultra"  # M Max/Ultra
        elif cpu >= 12 and mem >= 32:
            return "max"  # M Pro 高配
        elif cpu >= 10 and mem >= 16:
            return "pro"  # M Pro 基础
        else:
            return "basic"  # M1/M2 基础版
    
    @staticmethod
    def calculate_concurrency_limits(hardware: Dict[str, Any]) -> Dict[str, Any]:
        """计算各阶段的并发限制"""
        cpu = hardware["cpu_count"]
        mem = hardware["memory_gb"]
        
        # MD5 并发：IO 密集型，不是性能瓶颈
        md5_min = 1
        md5_max = min(4, cpu // 2)
        md5_recommended = min(2, cpu // 4)
        
        # 场景检测并发：CPU 密集型，每任务需要 2-3 核
        # 性能均衡：留一半 CPU 给系统
        # 极致性能：使用 75% CPU
        sifting_min = 1
        sifting_max = max(2, int(cpu * 0.75) // 2)  # 每任务2核，用75% CPU
        sifting_recommended = max(1, int(cpu * 0.5) // 2)  # 每任务2核，用50% CPU
        
        # VLM 并发：GPU/内存密集型（Ollama 运行）
        # 每个实例需要约 12GB 内存
        # 性能均衡：最多 2 个（留资源给系统）
        # 极致性能：根据内存计算，但预留 20GB 给系统
        vlm_min = 1
        available_mem_max = mem - 20  # 预留 20GB 给系统
        available_mem_rec = mem - 40  # 预留 40GB 给系统（性能均衡）
        vlm_max = max(1, min(4, int(available_mem_max / HardwareDetector.VLM_MEMORY_PER_INSTANCE)))
        vlm_recommended = max(1, min(2, int(available_mem_rec / HardwareDetector.VLM_MEMORY_PER_INSTANCE)))
        
        # 总并发：受限于最慢的阶段（通常是 VLM）
        total_min = 1
        total_max = max(2, min(cpu, int(mem / 8)))  # 每 8GB 支持 1 个任务
        total_recommended = max(1, min(total_max // 2, vlm_recommended * 2))  # 留一半余量
        
        return {
            "md5": {
                "min": md5_min,
                "max": md5_max,
                "recommended": md5_recommended,
                "description": "MD5 计算并发（IO 密集型）"
            },
            "sifting": {
                "min": sifting_min,
                "max": sifting_max,
                "recommended": sifting_recommended,
                "description": "场景检测+抽帧并发（CPU 密集型）"
            },
            "vlm": {
                "min": vlm_min,
                "max": vlm_max,
                "recommended": vlm_recommended,
                "description": "VLM 分析并发（GPU/内存密集型）"
            },
            "total": {
                "min": total_min,
                "max": total_max,
                "recommended": total_recommended,
                "description": "总任务并发上限"
            }
        }
    
    @staticmethod
    def generate_presets(limits: Dict[str, Any], device_class: str) -> Dict[str, Any]:
        """生成预设配置"""
        presets = {}
        
        # 极致节省：所有阶段使用最小值
        presets["economy"] = {
            "name": "极致节省",
            "description": "最低资源占用，适合后台任务",
            "max_concurrent": limits["total"]["min"],
            "max_md5_concurrent": limits["md5"]["min"],
            "max_sifting_concurrent": limits["sifting"]["min"],
            "max_vlm_concurrent": limits["vlm"]["min"]
        }
        
        # 性能均衡：使用推荐值
        presets["balanced"] = {
            "name": "性能均衡",
            "description": "平衡性能和资源占用，推荐日常使用",
            "max_concurrent": limits["total"]["recommended"],
            "max_md5_concurrent": limits["md5"]["recommended"],
            "max_sifting_concurrent": limits["sifting"]["recommended"],
            "max_vlm_concurrent": limits["vlm"]["recommended"]
        }
        
        # 极致性能：使用最大值
        presets["performance"] = {
            "name": "极致性能",
            "description": "最大化性能，可能影响系统响应",
            "max_concurrent": limits["total"]["max"],
            "max_md5_concurrent": limits["md5"]["max"],
            "max_sifting_concurrent": limits["sifting"]["max"],
            "max_vlm_concurrent": limits["vlm"]["max"]
        }
        
        return presets
    
    @staticmethod
    def get_recommendations() -> Dict[str, Any]:
        """获取完整推荐配置"""
        hardware = HardwareDetector.detect()
        
        # 检查最低要求
        requirement_check = HardwareDetector.check_minimum_requirements(hardware)
        
        # 设备分级
        device_class = HardwareDetector.classify_device(hardware)
        
        # 计算并发限制
        limits = HardwareDetector.calculate_concurrency_limits(hardware)
        
        # 生成预设
        presets = HardwareDetector.generate_presets(limits, device_class)
        
        # 设备等级描述
        class_descriptions = {
            "basic": "M1/M2 基础版（8-10 核，16GB）",
            "pro": "M Pro 系列（10-12 核，16-32GB）",
            "max": "M Max 系列（12-16 核，32-64GB）",
            "ultra": "M Ultra 系列（16+ 核，64GB+）"
        }
        
        return {
            "hardware": hardware,
            "device_class": device_class,
            "device_description": class_descriptions.get(device_class, "未知"),
            "requirement_check": requirement_check,
            "limits": {
                "max_concurrent": {
                    "min": limits["total"]["min"],
                    "max": limits["total"]["max"],
                    "recommended": limits["total"]["recommended"],
                    "description": limits["total"]["description"]
                },
                "max_md5_concurrent": {
                    "min": limits["md5"]["min"],
                    "max": limits["md5"]["max"],
                    "recommended": limits["md5"]["recommended"],
                    "description": limits["md5"]["description"]
                },
                "max_sifting_concurrent": {
                    "min": limits["sifting"]["min"],
                    "max": limits["sifting"]["max"],
                    "recommended": limits["sifting"]["recommended"],
                    "description": limits["sifting"]["description"]
                },
                "max_vlm_concurrent": {
                    "min": limits["vlm"]["min"],
                    "max": limits["vlm"]["max"],
                    "recommended": limits["vlm"]["recommended"],
                    "description": limits["vlm"]["description"]
                }
            },
            "presets": presets,
            "recommended": {
                "max_concurrent": limits["total"]["recommended"],
                "max_md5_concurrent": limits["md5"]["recommended"],
                "max_sifting_concurrent": limits["sifting"]["recommended"],
                "max_vlm_concurrent": limits["vlm"]["recommended"]
            }
        }
