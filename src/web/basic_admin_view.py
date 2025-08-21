"""
Basic Admin View for Step 7 - Thesis Version
Simple web interface for viewing attendance data
"""

from flask import Blueprint, render_template, request, jsonify, send_file
from datetime import datetime, timedelta
import json
import os
import logging

from ..logging.simple_attendance_logger import SimpleAttendanceLogger
from ..reports.basic_report_generator_simple import BasicReportGenerator

# Create simple logger
logger = logging.getLogger(__name__)


# Create blueprint for admin views
admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# Initialize components
attendance_logger = SimpleAttendanceLogger()
report_generator = BasicReportGenerator(attendance_logger)
web_logger = logger


@admin_bp.route('/')
def admin_dashboard():
    """
    Main admin dashboard
    """
    try:
        # Get today's statistics
        today = datetime.now().strftime('%Y-%m-%d')
        today_summary = attendance_logger.get_daily_attendance_summary(today)
        
        # Get recent attendance statistics
        stats = attendance_logger.get_attendance_statistics(days=7)
        
        # Get recent system events
        recent_events = attendance_logger.get_recent_system_events(limit=20)
        
        return render_template('admin/dashboard.html', 
                             today_summary=today_summary,
                             stats=stats,
                             recent_events=recent_events)
    
    except Exception as e:
        logger.error(f"Error in admin dashboard: {e}")
        return render_template('admin/error.html', error=str(e))


@admin_bp.route('/attendance/daily')
def daily_attendance():
    """
    View daily attendance records
    """
    try:
        date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        # Generate daily report
        daily_report = report_generator.generate_daily_report(date)
        
        return render_template('admin/daily_attendance.html', 
                             report=daily_report,
                             selected_date=date)
    
    except Exception as e:
        logger.error(f"Error in daily attendance view: {e}")
        return render_template('admin/error.html', error=str(e))


@admin_bp.route('/attendance/weekly')
def weekly_attendance():
    """
    View weekly attendance summary
    """
    try:
        start_date = request.args.get('start_date')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        # Generate weekly report
        weekly_report = report_generator.generate_weekly_report(start_date)
        
        return render_template('admin/weekly_attendance.html', 
                             report=weekly_report,
                             start_date=start_date)
    
    except Exception as e:
        logger.error(f"Error in weekly attendance view: {e}")
        return render_template('admin/error.html', error=str(e))


@admin_bp.route('/users')
def users_list():
    """
    View registered users
    """
    try:
        # Get all users from the main database
        from ..database.attendance_db import AttendanceDatabase
        db = AttendanceDatabase()
        users = db.get_all_users()
        
        # Get attendance stats for each user
        for user in users:
            user_history = attendance_logger.get_user_attendance_history(
                user['user_id'], days=30
            )
            user['recent_attendance_count'] = len([
                h for h in user_history if h['status'] == 'success'
            ])
            user['last_attendance'] = user_history[0]['timestamp'] if user_history else None
        
        return render_template('admin/users.html', users=users)
    
    except Exception as e:
        logger.error(f"Error in users list: {e}")
        return render_template('admin/error.html', error=str(e))


@admin_bp.route('/user/<user_id>')
def user_detail(user_id):
    """
    View detailed user information
    """
    try:
        # Get user info
        from ..database.attendance_db import AttendanceDatabase
        db = AttendanceDatabase()
        user_info = db.get_user_info(user_id)
        
        if not user_info:
            return render_template('admin/error.html', error="User not found")
        
        # Get attendance history
        attendance_history = attendance_logger.get_user_attendance_history(
            user_id, days=30
        )
        
        # Calculate user statistics
        successful_attendance = len([h for h in attendance_history if h['status'] == 'success'])
        total_attempts = len(attendance_history)
        success_rate = (successful_attendance / max(total_attempts, 1)) * 100
        
        user_stats = {
            'successful_attendance': successful_attendance,
            'total_attempts': total_attempts,
            'success_rate': round(success_rate, 2),
            'avg_confidence': round(sum(h['confidence_score'] or 0 for h in attendance_history) / max(total_attempts, 1), 3)
        }
        
        return render_template('admin/user_detail.html', 
                             user=user_info,
                             attendance_history=attendance_history,
                             user_stats=user_stats)
    
    except Exception as e:
        logger.error(f"Error in user detail: {e}")
        return render_template('admin/error.html', error=str(e))


@admin_bp.route('/reports')
def reports_page():
    """
    Reports generation page
    """
    return render_template('admin/reports.html')


@admin_bp.route('/api/generate_report', methods=['POST'])
def generate_report():
    """
    API endpoint to generate reports
    """
    try:
        data = request.get_json()
        report_type = data.get('report_type', 'daily')
        date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        if report_type == 'daily':
            report = report_generator.generate_daily_report(date)
        elif report_type == 'weekly':
            start_date = data.get('start_date', 
                                (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))
            report = report_generator.generate_weekly_report(start_date)
        else:
            return jsonify({'error': 'Invalid report type'}), 400
        
        return jsonify(report)
    
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/api/export_data', methods=['POST'])
def export_data():
    """
    API endpoint to export attendance data
    """
    try:
        data = request.get_json()
        export_type = data.get('export_type', 'csv')
        days = data.get('days', 30)
        
        if export_type == 'thesis':
            # Generate comprehensive thesis export
            exports = report_generator.generate_thesis_data_export(days)
            return jsonify({
                'success': True,
                'exports': exports,
                'message': f'Generated {len(exports)} export files'
            })
        
        else:
            # Export attendance data
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            attendance_data = attendance_logger.export_attendance_data(start_date=start_date)
            
            if export_type == 'csv':
                filepath = report_generator.export_to_csv(attendance_data)
            elif export_type == 'json':
                filepath = report_generator.export_to_json(attendance_data)
            else:
                return jsonify({'error': 'Invalid export type'}), 400
            
            return jsonify({
                'success': True,
                'filepath': filepath,
                'message': f'Data exported to {filepath}'
            })
    
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/api/statistics')
def api_statistics():
    """
    API endpoint to get attendance statistics
    """
    try:
        days = request.args.get('days', 30, type=int)
        stats = attendance_logger.get_attendance_statistics(days)
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/download/<filename>')
def download_file(filename):
    """
    Download exported files
    """
    try:
        filepath = report_generator.output_dir / filename
        if filepath.exists():
            return send_file(str(filepath), as_attachment=True)
        else:
            return "File not found", 404
    
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return "Error downloading file", 500


@admin_bp.route('/system_status')
def system_status():
    """
    View system status and health
    """
    try:
        # Get database statistics
        from ..database.attendance_db import AttendanceDatabase
        db = AttendanceDatabase()
        db_stats = db.get_database_stats()
        
        # Get recent system events
        recent_events = attendance_logger.get_recent_system_events(limit=50)
        
        # Get system health metrics
        today = datetime.now().strftime('%Y-%m-%d')
        today_summary = attendance_logger.get_daily_attendance_summary(today)
        
        # Calculate system health score
        success_rate = today_summary.get('success_rate', 0)
        total_attempts = today_summary.get('total_attempts', 0)
        
        if total_attempts == 0:
            health_score = 100  # No attempts, system idle
        elif success_rate >= 95:
            health_score = 100
        elif success_rate >= 90:
            health_score = 90
        elif success_rate >= 80:
            health_score = 80
        else:
            health_score = 60
        
        system_status = {
            'health_score': health_score,
            'status': 'Healthy' if health_score >= 90 else 'Warning' if health_score >= 80 else 'Critical',
            'database_stats': db_stats,
            'today_summary': today_summary,
            'recent_events': recent_events
        }
        
        return render_template('admin/system_status.html', 
                             system_status=system_status)
    
    except Exception as e:
        logger.error(f"Error in system status: {e}")
        return render_template('admin/error.html', error=str(e))


# Helper function to register the blueprint
def register_admin_routes(app):
    """
    Register admin routes with the Flask app
    
    Args:
        app: Flask application instance
    """
    app.register_blueprint(admin_bp)
    
    # Create templates directory if it doesn't exist
    templates_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'templates', 'admin')
    os.makedirs(templates_dir, exist_ok=True)
    
    logger.info("Admin routes registered successfully")
