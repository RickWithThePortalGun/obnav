# Remote Camera Functionality

This document describes the new remote camera functionality that allows using a server's camera instead of the Raspberry Pi camera when the remote detector is active.

## Overview

The remote camera feature enables the Raspberry Pi to use a server's camera for obstacle detection when the Pi's camera is faulty or unavailable. This is particularly useful for testing and development scenarios where you want to maintain the same detection pipeline but use a different camera source.

## Components

### 1. RemoteCamera Class (`run_prod_visual_enhanced.py`)
- Captures video frames from a remote server via ZMQ
- Handles connection management and error recovery
- Provides the same interface as the local PiCamera class

### 2. Camera Server (`camera_server.py`)
- Streams video from the server's camera to remote clients
- Uses ZMQ PUB/SUB pattern for efficient streaming
- Configurable resolution, FPS, and quality settings

### 3. Setup Script (`setup_remote_camera.py`)
- Helps configure the remote camera setup
- Tests camera accessibility and dependencies
- Provides usage instructions

## Usage

### Method 1: Environment Variables (Recommended)

**On the Server (laptop/desktop with camera):**
```bash
# Start the camera streaming server
python camera_server.py

# Start the detection server (in another terminal)
python server.py
```

**On the Raspberry Pi:**
```bash
# Set environment variables
export USE_REMOTE_CAMERA=1
export HOST_IP=192.168.1.100  # Replace with server IP
export SERVER_PORT=5555

# Run the enhanced script
python run_prod_visual_enhanced.py
```

### Method 2: Interactive Mode

**On the Raspberry Pi:**
```bash
# Run without environment variables
python run_prod_visual_enhanced.py

# You'll be prompted to:
# 1. Choose camera mode (Local/Remote)
# 2. Enter server IP address
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_REMOTE_CAMERA` | `0` | Set to `1` to enable remote camera mode |
| `HOST_IP` | `172.20.10.10` | IP address of the remote server |
| `SERVER_PORT` | `5555` | Port for detection server |
| `CAMERA_PORT` | `5556` | Port for camera streaming |
| `CAMERA_INDEX` | `0` | Camera index on server (0 for first camera) |
| `CAMERA_FPS` | `15` | Frames per second for camera streaming |
| `CAMERA_WIDTH` | `640` | Camera resolution width |
| `CAMERA_HEIGHT` | `480` | Camera resolution height |

### Camera Server Options

The camera server supports the following environment variables:

```bash
CAMERA_INDEX=0 CAMERA_PORT=5556 CAMERA_FPS=15 python camera_server.py
```

## Network Requirements

- **Camera Streaming**: Server port 5556 (configurable)
- **Detection Server**: Server port 5555 (configurable)
- **Protocol**: ZMQ over TCP
- **Bandwidth**: ~1-2 Mbps for 640x480@15fps

## Error Handling

The system includes comprehensive error handling:

1. **Camera Connection Failures**: Falls back to local camera if remote camera fails
2. **Network Issues**: Automatic reconnection with exponential backoff
3. **Server Unavailability**: Graceful degradation to local detection
4. **Frame Decoding Errors**: Returns black frames as fallback

## Troubleshooting

### Common Issues

1. **"Failed to connect to remote camera"**
   - Check if camera server is running on the server
   - Verify IP address and port
   - Check firewall settings

2. **"No frame received from remote camera"**
   - Camera server may not be running
   - Network connectivity issues
   - Camera may be in use by another application

3. **"Remote detector failed"**
   - Detection server not running
   - Wrong server IP or port
   - Network connectivity issues

### Debug Steps

1. **Test camera server:**
   ```bash
   python camera_server.py
   # Should show "Camera server ready" and preview window
   ```

2. **Test detection server:**
   ```bash
   python server.py
   # Should show "Obnav split-compute server" message
   ```

3. **Test connectivity from Pi:**
   ```bash
   # Test camera connection
   python -c "from run_prod_visual_enhanced import RemoteCamera; cam = RemoteCamera('SERVER_IP'); print('Camera test:', cam.read().shape)"
   
   # Test detection connection
   python -c "from adapters_prod_remote import RemoteDetector; det = RemoteDetector('SERVER_IP'); print('Detection test:', det.test_connection())"
   ```

## Performance Considerations

- **Latency**: Remote camera adds ~50-100ms latency
- **Bandwidth**: ~1-2 Mbps for camera streaming
- **CPU Usage**: Minimal impact on Pi, moderate on server
- **Memory**: ~10-20MB additional memory usage

## Security Notes

- ZMQ connections are unauthenticated
- Use only on trusted networks
- Consider VPN for remote access
- Firewall configuration may be required

## Integration with Existing System

The remote camera functionality is designed to be:
- **Transparent**: Same interface as local camera
- **Toggleable**: Can be enabled/disabled easily
- **Backward Compatible**: Doesn't affect existing local operation
- **Fallback Ready**: Automatically falls back to local camera if remote fails

## Example Workflows

### Development Testing
```bash
# On development machine
python camera_server.py &
python server.py &

# On Pi (for testing)
export USE_REMOTE_CAMERA=1
export HOST_IP=192.168.1.50
python run_prod_visual_enhanced.py
```

### Production with Faulty Pi Camera
```bash
# On server
python camera_server.py &
python server.py &

# On Pi
export USE_REMOTE_CAMERA=1
export HOST_IP=SERVER_IP
python run_prod_visual_enhanced.py
```

### Quick Setup
```bash
# Run setup script for guided configuration
python setup_remote_camera.py
```

## Files Modified/Created

- **Modified**: `run_prod_visual_enhanced.py` - Added RemoteCamera class and toggle functionality
- **Created**: `camera_server.py` - Camera streaming server
- **Created**: `setup_remote_camera.py` - Setup and testing script
- **Created**: `REMOTE_CAMERA_README.md` - This documentation

## Future Enhancements

Potential improvements for the remote camera system:
- Authentication for ZMQ connections
- Compression options for bandwidth optimization
- Multiple camera support
- Web-based configuration interface
- Automatic server discovery
- Quality adaptation based on network conditions
