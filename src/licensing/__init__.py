"""MediaTag 授权模块（集成依据：license-system/docs/MediaTag客户端集成规范.md v1.0）。

组件：fingerprint（设备指纹）/ client（API+Ed25519 验签）/ store（Keychain 存储）/
gate（启动门禁）/ activation_dialog（激活与锁定窗）/ status_panel（设置页授权区）。

安全红线（规范 §9）：无任何"开发模式跳过授权"路径；API_BASE 与公钥硬编码；
验签失败的响应一律按不可信处理；网络层只访问 API_BASE 单一域名。
"""
