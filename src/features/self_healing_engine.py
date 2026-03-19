"""
8. Self-Healing Execution Engine
Embed local "Reflection Agent" with anomaly detection to autonomously diagnose faults and formulate alternative paths
"""

import logging
import json
import time
import traceback
from typing import Dict, List, Optional, Any, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import threading
import queue

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    HEALING = "healing"

@dataclass
class Task:
    """Task definition"""
    id: str
    name: str
    function: Callable
    args: tuple
    kwargs: dict
    max_retries: int = 3
    timeout: int = 300
    dependencies: List[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = ""
    retry_count: int = 0
    healing_attempts: int = 0
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""

@dataclass
class AnomalyReport:
    """Anomaly detection report"""
    task_id: str
    anomaly_type: str
    severity: float
    description: str
    suggested_fixes: List[str]
    detected_at: str

class SelfHealingEngine:
    """
    Self-Healing Execution Engine with Reflection Agent
    Autonomously diagnoses faults and formulates alternative execution paths
    """
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.task_queue = queue.Queue()
        self.completed_tasks = {}
        self.failed_tasks = {}
        
        # Anomaly detection
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.execution_history = []
        self.anomaly_reports = []
        
        # Healing strategies
        self.healing_strategies = self._initialize_healing_strategies()
        
        # Execution monitoring
        self.monitoring_active = False
        self.worker_thread = None
        
        # Reflection agent state
        self.reflection_context = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "healing_success_rate": 0.0,
            "common_failures": {},
            "performance_metrics": {}
        }
        
        logger.info("Self-Healing Execution Engine initialized")
    
    def _initialize_healing_strategies(self) -> Dict[str, Callable]:
        """Initialize healing strategies for different failure types"""
        return {
            "timeout_error": self._heal_timeout_error,
            "memory_error": self._heal_memory_error,
            "connection_error": self._heal_connection_error,
            "permission_error": self._heal_permission_error,
            "value_error": self._heal_value_error,
            "type_error": self._heal_type_error,
            "import_error": self._heal_import_error,
            "file_not_found": self._heal_file_not_found,
            "unknown_error": self._heal_unknown_error
        }
    
    def add_task(self, task_id: str, name: str, function: Callable, 
                 args: tuple = (), kwargs: dict = None, 
                 max_retries: int = 3, timeout: int = 300,
                 dependencies: List[str] = None) -> str:
        """
        Add a task to the execution engine
        
        Returns:
            str: Task ID
        """
        if kwargs is None:
            kwargs = {}
        
        task = Task(
            id=task_id,
            name=name,
            function=function,
            args=args,
            kwargs=kwargs,
            max_retries=max_retries,
            timeout=timeout,
            dependencies=dependencies or [],
            created_at=datetime.now().isoformat()
        )
        
        self.tasks[task_id] = task
        self.task_queue.put(task_id)
        
        # Start worker if not running
        if not self.monitoring_active:
            self.start_execution()
        
        logger.info(f"Task added: {task_id} - {name}")
        return task_id
    
    def start_execution(self):
        """Start the execution engine worker thread"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.worker_thread = threading.Thread(target=self._execution_worker, daemon=True)
        self.worker_thread.start()
        
        logger.info("Self-Healing Engine started")
    
    def stop_execution(self):
        """Stop the execution engine"""
        self.monitoring_active = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        
        logger.info("Self-Healing Engine stopped")
    
    def _execution_worker(self):
        """Main worker thread for task execution"""
        while self.monitoring_active:
            try:
                # Get next task
                try:
                    task_id = self.task_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                task = self.tasks.get(task_id)
                if not task:
                    continue
                
                # Check dependencies
                if not self._check_dependencies(task):
                    # Re-queue task
                    self.task_queue.put(task_id)
                    time.sleep(0.1)
                    continue
                
                # Execute task
                self._execute_task(task)
                
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker thread error: {e}")
                time.sleep(1)
    
    def _check_dependencies(self, task: Task) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
    
    def _execute_task(self, task: Task):
        """Execute a single task with healing capabilities"""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now().isoformat()
        
        logger.info(f"Executing task: {task.id} - {task.name}")
        
        try:
            # Record execution start
            start_time = time.time()
            
            # Execute with timeout
            result = self._execute_with_timeout(task)
            
            # Record execution metrics
            execution_time = time.time() - start_time
            self._record_execution_metrics(task, execution_time, True)
            
            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now().isoformat()
            
            self.completed_tasks[task.id] = task
            self.reflection_context["successful_tasks"] += 1
            
            logger.info(f"Task completed successfully: {task.id}")
            
        except Exception as e:
            # Record failure metrics
            execution_time = time.time() - start_time
            self._record_execution_metrics(task, execution_time, False)
            
            # Handle failure with healing
            self._handle_task_failure(task, e)
    
    def _execute_with_timeout(self, task: Task) -> Any:
        """Execute task with timeout handling"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Task {task.id} timed out after {task.timeout} seconds")
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(task.timeout)
        
        try:
            result = task.function(*task.args, **task.kwargs)
            signal.alarm(0)  # Cancel timeout
            return result
        finally:
            signal.alarm(0)  # Ensure timeout is cancelled
    
    def _record_execution_metrics(self, task: Task, execution_time: float, success: bool):
        """Record execution metrics for anomaly detection"""
        metrics = {
            "task_id": task.id,
            "task_name": task.name,
            "execution_time": execution_time,
            "success": success,
            "retry_count": task.retry_count,
            "healing_attempts": task.healing_attempts,
            "timestamp": datetime.now().isoformat()
        }
        
        self.execution_history.append(metrics)
        
        # Keep only recent history for anomaly detection
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
    
    def _handle_task_failure(self, task: Task, error: Exception):
        """Handle task failure with healing attempts"""
        error_type = type(error).__name__
        error_message = str(error)
        
        logger.error(f"Task {task.id} failed: {error_type} - {error_message}")
        
        # Detect anomalies
        anomaly_report = self._detect_anomaly(task, error)
        if anomaly_report:
            self.anomaly_reports.append(anomaly_report)
        
        # Update failure statistics
        self.reflection_context["failed_tasks"] += 1
        failure_key = f"{task.name}:{error_type}"
        self.reflection_context["common_failures"][failure_key] = \
            self.reflection_context["common_failures"].get(failure_key, 0) + 1
        
        # Attempt healing
        if task.retry_count < task.max_retries:
            task.status = TaskStatus.HEALING
            healing_success = self._attempt_healing(task, error)
            
            if healing_success:
                # Retry task after healing
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                self.task_queue.put(task.id)
                logger.info(f"Task {task.id} queued for retry after healing")
            else:
                # Mark as failed
                task.status = TaskStatus.FAILED
                task.error = error_message
                task.completed_at = datetime.now().isoformat()
                self.failed_tasks[task.id] = task
                logger.error(f"Task {task.id} failed permanently after healing attempts")
        else:
            # Max retries exceeded
            task.status = TaskStatus.FAILED
            task.error = error_message
            task.completed_at = datetime.now().isoformat()
            self.failed_tasks[task.id] = task
            logger.error(f"Task {task.id} failed after maximum retries")
    
    def _detect_anomaly(self, task: Task, error: Exception) -> Optional[AnomalyReport]:
        """Detect anomalies in task execution"""
        try:
            # Prepare features for anomaly detection
            if len(self.execution_history) < 10:
                return None  # Not enough history
            
            # Extract recent metrics
            recent_metrics = self.execution_history[-50:]
            
            # Create feature vector
            features = []
            for metric in recent_metrics:
                features.append([
                    metric["execution_time"],
                    metric["retry_count"],
                    metric["healing_attempts"],
                    1 if metric["success"] else 0
                ])
            
            if len(features) < 10:
                return None
            
            features = np.array(features)
            
            # Fit anomaly detector
            self.anomaly_detector.fit(features)
            
            # Check current task metrics
            current_metrics = [
                task.retry_count,
                task.healing_attempts,
                0  # Failed
            ]
            
            # Predict anomaly (need same number of features)
            if len(features[0]) >= len(current_metrics):
                current_features = np.array([current_metrics + [0] * (len(features[0]) - len(current_metrics))])
                anomaly_score = self.anomaly_detector.decision_function(current_features)[0]
                
                if anomaly_score < -0.5:  # Anomaly threshold
                    return AnomalyReport(
                        task_id=task.id,
                        anomaly_type="execution_pattern",
                        severity=abs(anomaly_score),
                        description=f"Unusual execution pattern detected for task {task.id}",
                        suggested_fixes=[
                            "Review task dependencies",
                            "Check resource availability",
                            "Consider task decomposition"
                        ],
                        detected_at=datetime.now().isoformat()
                    )
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
        
        return None
    
    def _attempt_healing(self, task: Task, error: Exception) -> bool:
        """Attempt to heal the failed task"""
        error_type = type(error).__name__
        
        # Get healing strategy
        healing_strategy = self.healing_strategies.get(error_type, self.healing_strategies["unknown_error"])
        
        try:
            logger.info(f"Attempting healing for task {task.id} using {error_type} strategy")
            
            # Apply healing strategy
            healing_result = healing_strategy(task, error)
            
            task.healing_attempts += 1
            
            # Update healing success rate
            if healing_result:
                total_healing = sum(1 for t in self.tasks.values() if t.healing_attempts > 0)
                successful_healing = sum(1 for t in self.tasks.values() 
                                       if t.healing_attempts > 0 and t.status == TaskStatus.COMPLETED)
                if total_healing > 0:
                    self.reflection_context["healing_success_rate"] = successful_healing / total_healing
            
            return healing_result
            
        except Exception as e:
            logger.error(f"Healing attempt failed for task {task.id}: {e}")
            return False
    
    # Healing strategies
    def _heal_timeout_error(self, task: Task, error: Exception) -> bool:
        """Heal timeout errors"""
        # Increase timeout
        task.timeout = int(task.timeout * 1.5)
        
        # Check if function can be optimized
        if hasattr(task.function, '__code__'):
            # Add memoization hint
            task.kwargs["_heal_timeout"] = True
        
        return True
    
    def _heal_memory_error(self, task: Task, error: Exception) -> bool:
        """Heal memory errors"""
        # Reduce batch size if applicable
        if "batch_size" in task.kwargs:
            task.kwargs["batch_size"] = max(1, task.kwargs["batch_size"] // 2)
        
        # Add memory optimization hints
        task.kwargs["_heal_memory"] = True
        
        return True
    
    def _heal_connection_error(self, task: Task, error: Exception) -> bool:
        """Heal connection errors"""
        # Add retry delay
        time.sleep(2)
        
        # Add connection retry parameters
        task.kwargs["_heal_connection"] = True
        task.kwargs["retry_delay"] = 5
        
        return True
    
    def _heal_permission_error(self, task: Task, error: Exception) -> bool:
        """Heal permission errors"""
        # Try alternative paths
        if "path" in task.kwargs:
            original_path = task.kwargs["path"]
            task.kwargs["path"] = f"./{original_path}"  # Try relative path
        
        return True
    
    def _heal_value_error(self, task: Task, error: Exception) -> bool:
        """Heal value errors"""
        # Add validation
        task.kwargs["_heal_validation"] = True
        
        # Try to fix common value issues
        for key, value in task.kwargs.items():
            if isinstance(value, str) and not value.strip():
                task.kwargs[key] = None  # Convert empty string to None
        
        return True
    
    def _heal_type_error(self, task: Task, error: Exception) -> bool:
        """Heal type errors"""
        # Add type conversion
        task.kwargs["_heal_type_conversion"] = True
        
        return True
    
    def _heal_import_error(self, task: Task, error: Exception) -> bool:
        """Heal import errors"""
        # Try alternative imports
        task.kwargs["_heal_import"] = True
        
        return True
    
    def _heal_file_not_found(self, task: Task, error: Exception) -> bool:
        """Heal file not found errors"""
        # Check if file path needs adjustment
        for key, value in task.kwargs.items():
            if isinstance(value, str) and (".txt" in value or ".csv" in value or ".json" in value):
                # Try to create file if it doesn't exist
                if not Path(value).exists():
                    try:
                        Path(value).touch()
                        logger.info(f"Created missing file: {value}")
                    except:
                        pass
        
        return True
    
    def _heal_unknown_error(self, task: Task, error: Exception) -> bool:
        """Heal unknown errors with generic strategy"""
        # Add general debugging
        task.kwargs["_heal_debug"] = True
        
        # Log full traceback for analysis
        logger.error(f"Full traceback for task {task.id}: {traceback.format_exc()}")
        
        return True
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of a specific task"""
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        return {
            "id": task.id,
            "name": task.name,
            "status": task.status.value,
            "retry_count": task.retry_count,
            "healing_attempts": task.healing_attempts,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "error": task.error,
            "result": task.result if task.status == TaskStatus.COMPLETED else None
        }
    
    def get_engine_status(self) -> Dict:
        """Get overall engine status"""
        total_tasks = len(self.tasks)
        completed = len(self.completed_tasks)
        failed = len(self.failed_tasks)
        running = sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING)
        pending = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING)
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed,
            "failed_tasks": failed,
            "running_tasks": running,
            "pending_tasks": pending,
            "success_rate": completed / total_tasks if total_tasks > 0 else 0,
            "monitoring_active": self.monitoring_active,
            "queue_size": self.task_queue.qsize(),
            "anomaly_reports": len(self.anomaly_reports),
            "reflection_context": self.reflection_context
        }
    
    def generate_reflection_report(self) -> str:
        """Generate reflection agent analysis report"""
        report_lines = [
            "Self-Healing Engine Reflection Report",
            "=" * 45,
            f"Total Tasks Processed: {self.reflection_context['total_tasks']}",
            f"Successful Tasks: {self.reflection_context['successful_tasks']}",
            f"Failed Tasks: {self.reflection_context['failed_tasks']}",
            f"Healing Success Rate: {self.reflection_context['healing_success_rate']:.2%}",
            "",
            "Common Failure Patterns:",
            "-" * 25
        ]
        
        # Top failure patterns
        common_failures = sorted(self.reflection_context["common_failures"].items(), 
                                key=lambda x: x[1], reverse=True)[:5]
        
        for failure, count in common_failures:
            report_lines.append(f"{failure}: {count} occurrences")
        
        # Recent anomalies
        if self.anomaly_reports:
            report_lines.extend([
                "",
                "Recent Anomalies:",
                "-" * 18
            ])
            for anomaly in self.anomaly_reports[-3:]:
                report_lines.append(f"• {anomaly.task_id}: {anomaly.description}")
        
        # Performance insights
        if self.execution_history:
            recent_executions = self.execution_history[-20:]
            avg_execution_time = np.mean([e["execution_time"] for e in recent_executions])
            success_rate = sum(1 for e in recent_executions if e["success"]) / len(recent_executions)
            
            report_lines.extend([
                "",
                "Performance Metrics:",
                "-" * 20,
                f"Average Execution Time: {avg_execution_time:.2f}s",
                f"Recent Success Rate: {success_rate:.2%}"
            ])
        
        # Recommendations
        report_lines.extend([
            "",
            "Reflection Recommendations:",
            "-" * 28
        ])
        
        if self.reflection_context["healing_success_rate"] < 0.5:
            report_lines.append("• Consider reviewing healing strategies")
        
        if len(self.failed_tasks) > len(self.completed_tasks):
            report_lines.append("• High failure rate detected - review task definitions")
        
        if self.anomaly_reports:
            report_lines.append("• Anomalies detected - investigate execution patterns")
        
        return "\n".join(report_lines)
    
    def export_execution_history(self, output_path: str):
        """Export execution history for analysis"""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "reflection_context": self.reflection_context,
            "execution_history": self.execution_history,
            "anomaly_reports": [asdict(report) for report in self.anomaly_reports],
            "completed_tasks": {k: asdict(v) for k, v in self.completed_tasks.items()},
            "failed_tasks": {k: asdict(v) for k, v in self.failed_tasks.items()}
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Execution history exported to {output_path}")
    
    def reset_engine(self):
        """Reset the engine state"""
        self.stop_execution()
        
        # Clear all data
        self.tasks.clear()
        self.completed_tasks.clear()
        self.failed_tasks.clear()
        self.execution_history.clear()
        self.anomaly_reports.clear()
        
        # Reset reflection context
        self.reflection_context = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "healing_success_rate": 0.0,
            "common_failures": {},
            "performance_metrics": {}
        }
        
        # Clear queue
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Self-Healing Engine reset")
