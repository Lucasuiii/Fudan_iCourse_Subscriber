"""Explicit course scan outcome; identifiers stay out of public audits."""
from dataclasses import dataclass, field


class CourseEnumerationError(RuntimeError):
    def __init__(self):
        super().__init__('Subscribed course enumeration incomplete')


@dataclass
class EnumerationResult:
    lectures: list = field(default_factory=list)
    successful_courses: list[str] = field(default_factory=list)
    failed_courses: list[str] = field(default_factory=list)

    def public_audit(self):
        return {'schema': 1, 'status': 'degraded' if self.failed_courses else 'complete',
                'successful_course_count': len(self.successful_courses),
                'failed_course_count': len(self.failed_courses)}

    def require_success(self):
        if self.failed_courses:
            raise CourseEnumerationError()
