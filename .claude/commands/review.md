---
description: Review code changes for quality
allowed-tools: Bash(git:*), Read, Grep
---

Review the changed files for quality assurance:

---

**Uncommitted files:**
!`git diff --stat --diff-filter=AM HEAD`

**Unpushed files:**
!`git diff --stat --diff-filter=AM @{u}...HEAD 2>/dev/null || echo "No upstream branch"`

**All changed files in this branch:**
!`git diff --name-only --diff-filter=AM $(git merge-base HEAD main) 2>/dev/null || git diff --name-only HEAD~5`

---

## Code Review Checklist

**1. Code Quality:**
- Readability and maintainability
- Consistent style and formatting
- Appropriate abstraction levels
- No code duplication (DRY principle)

**2. Python-Specific:**
- Type annotations present
- Pydantic models for structured data
- Fail-fast validation at boundaries
- Proper logging statements
- No silent exception swallowing

**3. Potential Issues:**
- Logic errors or bugs
- Edge cases not handled
- Performance concerns
- Security vulnerabilities

**4. Documentation:**
- Docstrings for public functions
- Updated README if API changed
- Inline comments only for complex logic

**5. Testing:**
- Tests cover new functionality
- Tests verify expected behavior (not just code paths)

Provide specific feedback with file and line references.
