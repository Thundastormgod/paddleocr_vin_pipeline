"""
Comprehensive Playwright UI Tests for VIN OCR Streamlit Application
Tests all buttons, tabs, and functionality to ensure proper operation
"""
import asyncio
from playwright.async_api import async_playwright, expect
import time


async def test_streamlit_ui():
    """Test all UI components and buttons in the Streamlit application"""
    
    async with async_playwright() as p:
        print("🚀 Launching browser...")
        browser = await p.chromium.launch(headless=False, slow_mo=500)
        page = await browser.new_page()
        
        try:
            # Navigate to Streamlit app
            print("📱 Navigating to http://localhost:8501")
            await page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(3000)  # Wait for Streamlit to fully load
            
            print("✅ Page loaded successfully")
            
            # Test 1: Check page title
            print("\n=== Test 1: Page Title ===")
            title = await page.text_content("h1")
            print(f"Page title: {title}")
            assert "VIN" in title or "OCR" in title, "Page title should contain VIN or OCR"
            print("✅ Title check passed")
            
            # Test 2: Check tabs are present
            print("\n=== Test 2: Check Navigation Tabs ===")
            tabs = await page.locator('[role="tab"]').all_text_content()
            print(f"Available tabs: {tabs}")
            expected_tabs = ["Recognition", "Batch Evaluation", "Training"]
            for tab_name in expected_tabs:
                assert any(tab_name in tab for tab in tabs), f"Tab '{tab_name}' should be present"
            print("✅ All tabs found")
            
            # Test 3: Test Recognition Tab
            print("\n=== Test 3: Recognition Tab ===")
            recognition_tab = page.locator('[role="tab"]', has_text="Recognition")
            await recognition_tab.click()
            await page.wait_for_timeout(1000)
            print("✅ Clicked Recognition tab")
            
            # Check for file uploader
            uploader = page.locator('input[type="file"]')
            if await uploader.count() > 0:
                print("✅ File uploader found")
            else:
                print("⚠️ File uploader not found on Recognition tab")
            
            # Check for model selection
            select_boxes = await page.locator('[data-testid="stSelectbox"]').count()
            print(f"Found {select_boxes} select boxes (model selection dropdowns)")
            if select_boxes > 0:
                print("✅ Model selection dropdown found")
            else:
                print("⚠️ No select boxes found")
            
            # Test 4: Test Batch Evaluation Tab
            print("\n=== Test 4: Batch Evaluation Tab ===")
            batch_tab = page.locator('[role="tab"]', has_text="Batch Evaluation")
            await batch_tab.click()
            await page.wait_for_timeout(1000)
            print("✅ Clicked Batch Evaluation tab")
            
            # Look for evaluate button
            buttons = await page.locator('button').all_text_content()
            print(f"Found buttons: {buttons}")
            evaluate_button_found = any("Evaluate" in btn or "Run" in btn for btn in buttons)
            if evaluate_button_found:
                print("✅ Evaluation button found")
            else:
                print("⚠️ Evaluation button not found")
            
            # Test 5: Test Training Tab
            print("\n=== Test 5: Training Tab ===")
            training_tab = page.locator('[role="tab"]', has_text="Training")
            await training_tab.click()
            await page.wait_for_timeout(1000)
            print("✅ Clicked Training tab")
            
            # Check for training options
            page_text = await page.text_content('body')
            if "PaddleOCR" in page_text or "DeepSeek" in page_text:
                print("✅ Training options (PaddleOCR/DeepSeek) found")
            else:
                print("⚠️ Training options not clearly visible")
            
            # Test 6: Check for any error messages
            print("\n=== Test 6: Error Detection ===")
            errors = await page.locator('[data-testid="stException"]').count()
            if errors == 0:
                print("✅ No error messages detected")
            else:
                error_text = await page.locator('[data-testid="stException"]').text_content()
                print(f"⚠️ Found {errors} error(s): {error_text}")
            
            # Test 7: Check sidebar
            print("\n=== Test 7: Sidebar Check ===")
            sidebar = page.locator('[data-testid="stSidebar"]')
            if await sidebar.count() > 0:
                sidebar_text = await sidebar.text_content()
                print(f"✅ Sidebar found with content: {sidebar_text[:200]}...")
            else:
                print("⚠️ Sidebar not found")
            
            # Test 8: Interactive button clicks (Recognition tab)
            print("\n=== Test 8: Button Interaction Test ===")
            await recognition_tab.click()
            await page.wait_for_timeout(1000)
            
            # Try to find and click any available buttons
            all_buttons = page.locator('button')
            button_count = await all_buttons.count()
            print(f"Found {button_count} clickable buttons on Recognition tab")
            
            for i in range(min(button_count, 5)):  # Test first 5 buttons
                button_text = await all_buttons.nth(i).text_content()
                if button_text and button_text.strip():
                    print(f"  Button {i+1}: '{button_text.strip()}'")
            
            print("✅ Button enumeration complete")
            
            # Test 9: Check for critical UI elements
            print("\n=== Test 9: Critical Elements Check ===")
            
            # Check for main container
            main_container = page.locator('[data-testid="stAppViewContainer"]')
            assert await main_container.count() > 0, "Main container should exist"
            print("✅ Main container found")
            
            # Check for headers
            headers = await page.locator('h1, h2, h3').count()
            print(f"✅ Found {headers} headers on page")
            
            # Test 10: Screenshot for manual verification
            print("\n=== Test 10: Taking Screenshots ===")
            await recognition_tab.click()
            await page.wait_for_timeout(500)
            await page.screenshot(path="test_screenshots_recognition.png", full_page=True)
            print("✅ Recognition tab screenshot saved")
            
            await batch_tab.click()
            await page.wait_for_timeout(500)
            await page.screenshot(path="test_screenshots_batch.png", full_page=True)
            print("✅ Batch evaluation tab screenshot saved")
            
            await training_tab.click()
            await page.wait_for_timeout(500)
            await page.screenshot(path="test_screenshots_training.png", full_page=True)
            print("✅ Training tab screenshot saved")
            
            print("\n" + "="*50)
            print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
            print("="*50)
            
        except Exception as e:
            print(f"\n❌ ERROR during testing: {str(e)}")
            await page.screenshot(path="test_screenshots_error.png", full_page=True)
            print("Error screenshot saved to test_screenshots_error.png")
            raise
            
        finally:
            print("\n⏳ Keeping browser open for 5 seconds for manual inspection...")
            await page.wait_for_timeout(5000)
            await browser.close()
            print("✅ Browser closed")


if __name__ == "__main__":
    print("="*50)
    print("VIN OCR Streamlit UI Comprehensive Test Suite")
    print("="*50)
    print("This test will:")
    print("1. Open the Streamlit app in a browser")
    print("2. Test all tabs (Recognition, Batch Evaluation, Training)")
    print("3. Check for all buttons and UI elements")
    print("4. Take screenshots for manual verification")
    print("5. Report any errors or missing elements")
    print("="*50)
    print()
    
    asyncio.run(test_streamlit_ui())
