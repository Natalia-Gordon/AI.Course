# 🚀 Production Deployment Guide

## 📋 Overview

This guide provides comprehensive instructions for deploying the Hybrid RAG System in production environments. The system is designed to handle financial document analysis with high reliability and performance.

## 🎯 Production Requirements

### System Requirements

- **CPU**: 4+ cores (8+ recommended)
- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 50GB+ SSD storage
- **Network**: Stable internet connection for API calls
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows 10+

### API Requirements

- **OpenAI API**: GPT-4 access with sufficient credits
- **Pinecone**: Production environment with appropriate tier
- **LlamaCloud**: API access for enhanced extraction (optional)

## 🔧 Pre-Deployment Checklist

### 1. Environment Setup

```bash
# Create production environment
python -m venv venv_prod
source venv_prod/bin/activate  # Linux/Mac
# or
venv_prod\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration Validation

```bash
# Validate configuration
python -c "from src.core.config_manager import ConfigManager; ConfigManager().validate_configuration()"

# Check environment variables
python -c "import os; print('OPENAI_API_KEY:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"
python -c "import os; print('PINECONE_API_KEY:', 'SET' if os.getenv('PINECONE_API_KEY') else 'NOT SET')"
```

### 3. System Validation

```bash
# Run system validation
python src/eval/run_evaluation.py

# Check logs
tail -f logs/ragas_evaluation_*.log
```

## 🚀 Deployment Options

### Option 1: Direct Deployment

```bash
# Clone repository
git clone <repository-url>
cd Final_Work

# Set environment variables
export OPENAI_API_KEY="your_key"
export PINECONE_API_KEY="your_key"
export PINECONE_ENVIRONMENT="your_env"

# Run system
python src/main.py --query "test query"
```

### Option 2: Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "src/main.py"]
```

```bash
# Build and run
docker build -t hybrid-rag .
docker run -e OPENAI_API_KEY=your_key -e PINECONE_API_KEY=your_key hybrid-rag
```

### Option 3: Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hybrid-rag
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hybrid-rag
  template:
    metadata:
      labels:
        app: hybrid-rag
    spec:
      containers:
      - name: hybrid-rag
        image: hybrid-rag:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-key
        - name: PINECONE_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: pinecone-key
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
```

## 📊 Monitoring and Logging

### Log Management

```bash
# Monitor logs in real-time
tail -f logs/ragas_evaluation_*.log

# Check log rotation
ls -la logs/

# Analyze log performance
grep "performance" logs/ragas_evaluation_*.log
```

### Performance Monitoring

```bash
# Run regular evaluations
python src/eval/run_evaluation.py

# Check metrics
python src/eval/show_results.py

# Monitor API usage
grep "API call" logs/ragas_evaluation_*.log
```

### Health Checks

```python
# health_check.py
import requests
import os

def check_system_health():
    checks = {
        'openai_api': check_openai_api(),
        'pinecone_connection': check_pinecone_connection(),
        'evaluation_system': check_evaluation_system(),
        'log_system': check_log_system()
    }
    return checks

def check_openai_api():
    try:
        # Test OpenAI API
        return True
    except:
        return False

# Add to monitoring system
```

## 🔒 Security Considerations

### API Key Management

```bash
# Use environment variables (recommended)
export OPENAI_API_KEY="your_key"

# Or use .env file (ensure it's not committed)
echo "OPENAI_API_KEY=your_key" > .env
echo ".env" >> .gitignore
```

### Network Security

- **Firewall**: Restrict access to necessary ports only
- **VPN**: Use VPN for secure connections
- **HTTPS**: Ensure all external communications use HTTPS
- **Rate Limiting**: Implement API rate limiting

### Data Privacy

- **Data Encryption**: Encrypt sensitive data at rest
- **Access Control**: Implement role-based access control
- **Audit Logging**: Log all data access and modifications
- **Data Retention**: Implement data retention policies

## 📈 Scaling Strategies

### Horizontal Scaling

```bash
# Load balancer configuration
upstream hybrid_rag_backend {
    server 192.168.1.10:8000;
    server 192.168.1.11:8000;
    server 192.168.1.12:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://hybrid_rag_backend;
    }
}
```

### Vertical Scaling

```bash
# Increase system resources
# Update config.yaml
retrieval:
  dense_k: 20      # Increase from 10
  sparse_k: 20     # Increase from 10
  final_k: 12      # Increase from 6
```

### Caching Strategy

```python
# Implement Redis caching
import redis

class CacheManager:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    def cache_query(self, query, results):
        self.redis_client.setex(f"query:{hash(query)}", 3600, results)
    
    def get_cached_query(self, query):
        return self.redis_client.get(f"query:{hash(query)}")
```

## 🚨 Troubleshooting

### Common Issues

1. **API Rate Limits**
   ```bash
   # Check API usage
   grep "rate limit" logs/ragas_evaluation_*.log
   
   # Implement backoff strategy
   ```

2. **Memory Issues**
   ```bash
   # Monitor memory usage
   top -p $(pgrep -f "python.*main.py")
   
   # Check for memory leaks
   ```

3. **Performance Degradation**
   ```bash
   # Run performance evaluation
   python src/eval/run_evaluation.py
   
   # Check metrics
   python src/eval/show_results.py
   ```

### Emergency Procedures

```bash
# System restart
sudo systemctl restart hybrid-rag

# Rollback to previous version
git checkout HEAD~1

# Emergency shutdown
pkill -f "python.*main.py"
```

## 📋 Maintenance Schedule

### Daily Tasks

- [ ] Check system logs for errors
- [ ] Monitor API usage and costs
- [ ] Verify system performance metrics

### Weekly Tasks

- [ ] Run full evaluation pipeline
- [ ] Review and rotate log files
- [ ] Update system dependencies
- [ ] Backup configuration and data

### Monthly Tasks

- [ ] Performance optimization review
- [ ] Security audit
- [ ] Cost analysis and optimization
- [ ] System health assessment

## 🔄 Backup and Recovery

### Backup Strategy

```bash
# Configuration backup
tar -czf config_backup_$(date +%Y%m%d).tar.gz src/config.yaml

# Data backup
tar -czf data_backup_$(date +%Y%m%d).tar.gz data/

# Log backup
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
```

### Recovery Procedures

```bash
# Restore configuration
tar -xzf config_backup_YYYYMMDD.tar.gz

# Restore data
tar -xzf data_backup_YYYYMMDD.tar.gz

# Restart system
python src/main.py
```

## 📞 Support and Maintenance

### Contact Information

- **Technical Support**: [support@company.com]
- **Emergency Contact**: [emergency@company.com]
- **Documentation**: [docs.company.com]

### Escalation Procedures

1. **Level 1**: System administrators
2. **Level 2**: Development team
3. **Level 3**: Architecture team
4. **Level 4**: Vendor support

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Status**: Production Ready ✅
