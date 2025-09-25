#!/usr/bin/env python3
"""
System test for Bank AI LLM with pre-built database.
Verifies that the system works end-to-end without manual setup.
"""

import requests
import json
import time
import sys


def test_api_health():
    """Test API health endpoint."""
    print("🔍 Testing API health...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ API healthy: {health_data.get('status')}")
            print(f"   ✅ Database: {health_data.get('database')}")
            print(f"   ✅ LLM Service: {health_data.get('llm_service')}")
            return True
        else:
            print(f"   ❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ API health check error: {e}")
        return False


def test_database_stats():
    """Test database statistics endpoint."""
    print("🔍 Testing database statistics...")
    try:
        response = requests.get("http://localhost:8000/stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            clients = stats.get('clients', 0)
            accounts = stats.get('accounts', 0)
            transactions = stats.get('transactions', 0)
            regions = stats.get('regions', [])

            print(f"   ✅ Clients: {clients:,}")
            print(f"   ✅ Accounts: {accounts:,}")
            print(f"   ✅ Transactions: {transactions:,}")
            print(f"   ✅ Regions: {len(regions)} ({', '.join(regions[:3])}...)")

            # Verify reasonable numbers
            if clients >= 10000 and accounts >= 15000 and transactions >= 500000:
                print("   ✅ Database has sufficient data for demo")
                return True
            else:
                print("   ❌ Database appears to have insufficient data")
                return False
        else:
            print(f"   ❌ Database stats failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Database stats error: {e}")
        return False


def test_sample_queries():
    """Test sample queries endpoint."""
    print("🔍 Testing sample queries...")
    try:
        response = requests.get("http://localhost:8000/samples", timeout=10)
        if response.status_code == 200:
            samples = response.json().get('samples', [])
            print(f"   ✅ {len(samples)} sample queries available")
            if samples:
                print(f"   ✅ Example: '{samples[0][:60]}...'")
            return len(samples) > 0
        else:
            print(f"   ❌ Sample queries failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Sample queries error: {e}")
        return False


def test_query_execution():
    """Test actual query execution."""
    print("🔍 Testing query execution...")
    try:
        query_data = {
            "query": "Show total clients by region",
            "export_format": "excel"
        }

        print("   📊 Executing test query...")
        response = requests.post(
            "http://localhost:8000/query",
            json=query_data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                sql_query = result.get('sql_query')
                results = result.get('results', [])
                export_path = result.get('export_path')

                print(f"   ✅ Query generated: {sql_query[:60]}...")
                print(f"   ✅ Results: {len(results)} rows")
                if export_path:
                    print(f"   ✅ Excel exported: {export_path}")

                return True
            else:
                print(f"   ❌ Query execution failed: {result.get('error')}")
                return False
        else:
            print(f"   ❌ Query request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Query execution error: {e}")
        return False


def main():
    """Run all system tests."""
    print("🏦 Bank AI LLM System Test")
    print("=========================")
    print("Testing pre-built database and instant startup...")
    print("")

    tests = [
        ("API Health", test_api_health),
        ("Database Stats", test_database_stats),
        ("Sample Queries", test_sample_queries),
        ("Query Execution", test_query_execution)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"🧪 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                print(f"   ✅ {test_name}: PASSED")
            else:
                print(f"   ❌ {test_name}: FAILED")
        except Exception as e:
            print(f"   ❌ {test_name}: ERROR - {e}")
        print("")

    print("📋 Test Summary")
    print(f"   Passed: {passed}/{total}")

    if passed == total:
        print("   🎉 ALL TESTS PASSED!")
        print("   🏦 System ready for banking evaluation!")
        return 0
    else:
        print("   ⚠️  Some tests failed")
        print("   🔧 Please check system status")
        return 1


if __name__ == "__main__":
    sys.exit(main())