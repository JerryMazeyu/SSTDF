from utils.file_namer import FileNamer
from dotenv import load_dotenv
import os
import paramiko
import time
import logging

logger = logging.getLogger(__name__)

class SSHClient:
    def __init__(self):
        # 加载环境变量
        load_dotenv()
        self.host = os.getenv('SSH_HOST')
        self.port = os.getenv('SSH_PORT')
        self.username = os.getenv('SSH_USERNAME')
        self.password = os.getenv('SSH_PASSWORD')
        self.remote_base_path = os.getenv('SSH_REMOTE_BASE_PATH').replace('\\', '/')
        self.remote_image_path = os.path.join(self.remote_base_path,'image').replace('\\', '/')
        self.remote_result_dir_path = os.path.join(self.remote_base_path,'outputs').replace('\\', '/')
        self.local_download_dir = "../results"
        self.process_id = self.get_process_id()

    def connect(self):
        """建立SSH连接"""
        try:
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(
                self.host,
                self.port,
                self.username,
                self.password
            )
            self.sftp = self.ssh.open_sftp()
            logger.info("成功连接到服务器")
        except Exception as e:
            logger.error(f"连接服务器失败: {str(e)}")
            raise
    
    def transfer_image_via_paramiko(self, local_image_path:str)->str:
        """
        通过paramiko传输图片到服务器
        Args:
            local_image_path: 本地图片路径
        Returns:
            str: 远程图片路径
        """
        # 获取local_image_path的文件类型
        self.file_type = os.path.splitext(local_image_path)[1]
        logger.info("本地文件类型", self.file_type)

        # 拼接remote_path
        image_name = self.process_id + self.file_type

        self.remote_image_path = os.path.join(self.remote_image_path, image_name).replace('\\', '/')

        logger.info("远程文件路径", self.remote_image_path)

        
        try:
            # 上传图片
            self.sftp.put(local_image_path, self.remote_image_path)
            
            logger.info(f"图片已成功传输到 {self.remote_image_path}")

            return self.remote_image_path

        except Exception as e:
            # 如果发生错误则关闭连接
            self.sftp.close()
            self.ssh.close()
            logger.error(f"传输失败: {e}")
            return None
        
    def download_result(self) -> tuple[str, str]:
        """
        从服务器下载结果文件
        Args:
            local_image_path: 本地图片路径
        Returns:
            str: 本地结果文件路径
        """
        # 本地下载地址
        local_result_file = os.path.join(self.local_download_dir, self.process_id)
        # 创建本地目录
        os.makedirs(os.path.join(local_result_file, 'preds'), exist_ok=True)
        os.makedirs(os.path.join(local_result_file, 'vis'), exist_ok=True)

        # 拼接远程结果文件路径和本地结果文件路径
        remote_json_result = os.path.join(self.remote_result_dir_path, 'preds', f"{self.process_id}.json").replace('\\', '/')
        remote_vis_result = os.path.join(self.remote_result_dir_path, 'vis', f"{self.process_id}{self.file_type}").replace('\\', '/')
        
        local_json_result = os.path.join(local_result_file, 'preds', f"{self.process_id}.json")
        local_vis_result = os.path.join(local_result_file, 'vis', f"{self.process_id}{self.file_type}")

        # 下载结果JSON文件和可视化文件
        try:
            logger.info(f"开始下载JSON结果: {remote_json_result} -> {local_json_result}")
            logger.info(f"开始下载可视化结果: {remote_vis_result} -> {local_vis_result}")
            self.sftp.get(remotepath=remote_json_result, localpath=local_json_result)
            self.sftp.get(remotepath=remote_vis_result, localpath=local_vis_result)

        except Exception as e:
            logger.error(f"下载结果失败: {e}")
            return None,None

        return local_json_result, local_vis_result

    def process_image(self, model_id: str, image_path: str):
        """
        处理图片
        Args:
            image_path: 本地图片路径
        Returns:
            str: 本地结果文件路径
        """
        try:
            # 连接服务器
            self.connect()

            # 上传图片
            remote_image_path = self.transfer_image_via_paramiko(image_path)

            # 执行Python命令,首先进入工作目录并激活conda环境
            cmd = f'''bash -l -c 'cd {self.remote_base_path} && \
source /home/zentek/miniconda3/etc/profile.d/conda.sh && \
conda activate openmmlab && \
/home/zentek/miniconda3/envs/openmmlab/bin/python3 -c "
from processer import process_chain
process_chain(\\"{remote_image_path}\\", \\"{model_id}\\", \\"{self.process_id}\\")"
' '''
            stdin, stdout, stderr = self.ssh.exec_command(cmd)
            
            # 获取输出
            result = stdout.read().decode().strip()
            error = stderr.read().decode().strip()
            
            logger.info(f"处理结果: {result}")
            logger.info(f"处理过程中出现错误: {error}")
                
            # 等待处理完成
            if not self.wait_for_processing_complete():
                return None
            else:
                # 下载结果文件
                pred_file_path, vis_file_path = self.download_result()
                
                return pred_file_path, vis_file_path

        finally:
            if self.sftp:
                self.sftp.close()
            if self.ssh:
                self.ssh.close()

    def wait_for_processing_complete(self,max_wait_time: int = 300) -> bool:
        """
        等待远程处理完成
        Args:
            max_wait_time: 最大等待时间（秒）
        Returns:
            bool: 是否处理完成
        """
        self.remote_result_dir_path = os.path.join(self.remote_result_dir_path, self.process_id).replace('\\', '/')
        start_time = time.time()
        
        while True:
            try:
                # 检查是否超时
                if time.time() - start_time > max_wait_time:
                    logger.error("等待处理完成超时")
                    return False
                    
                # 检查远程文件是否存在
                stdin, stdout, stderr = self.ssh.exec_command(f"ls {self.remote_result_dir_path}")
                if stderr.read().decode().strip():  # 如果有错误输出，说明文件不存在
                    logger.info("正在等待处理完成...")
                    time.sleep(3)  # 等待3秒后重试
                    continue
                return True  # 文件存在，返回成功
            except Exception as e:
                logger.error(f"检查远程文件时出错: {e}")
                time.sleep(3)
                continue

    def get_process_id(self)->str:
        """
        获取处理ID
        Returns:
            str: 处理ID
        """
        return FileNamer.generate_time_based_name()

