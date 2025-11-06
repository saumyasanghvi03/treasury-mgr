#!/bin/bash

echo "========================================================================"
echo "        Treasury Management System - Final Verification"
echo "========================================================================"
echo ""

# Check Python version
echo "1. Checking Python version..."
python --version
echo ""

# Check dependencies
echo "2. Checking dependencies..."
pip list | grep -E "streamlit|pandas|numpy|scikit-learn|plotly|pulp|scipy"
echo ""

# Run component tests
echo "3. Running component tests..."
python test_app.py
echo ""

# Run demo
echo "4. Running feature demo..."
python demo.py | tail -20
echo ""

# Run custom examples
echo "5. Running custom data examples..."
python examples_custom_data.py | tail -15
echo ""

# Check file structure
echo "6. Verifying file structure..."
echo "Main files:"
ls -lh *.py *.md *.txt 2>/dev/null | awk '{print "  " $9 " - " $5}'
echo ""
echo "Module files:"
ls -lh modules/*.py 2>/dev/null | awk '{print "  " $9 " - " $5}'
echo ""
echo "Utility files:"
ls -lh utils/*.py 2>/dev/null | awk '{print "  " $9 " - " $5}'
echo ""

# Count lines of code
echo "7. Code statistics..."
echo "Total Python lines:"
find . -name "*.py" -not -path "./.venv/*" -exec wc -l {} + | tail -1
echo ""

echo "========================================================================"
echo "                    Verification Complete ✓"
echo "========================================================================"
echo ""
echo "To run the application:"
echo "  streamlit run app.py"
echo ""
echo "To view documentation:"
echo "  - README.md - Overview and features"
echo "  - USAGE_GUIDE.md - Detailed usage instructions"
echo "  - examples_custom_data.py - Integration examples"
echo ""
