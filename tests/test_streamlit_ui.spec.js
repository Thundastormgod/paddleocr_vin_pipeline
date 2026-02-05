const { test, expect } = require('@playwright/test');

const BASE_URL = process.env.STREAMLIT_BASE_URL || 'http://localhost:8501';

test.describe('Streamlit VIN OCR UI', () => {
  test('loads the home page and core sections', async ({ page }) => {
    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });

    await expect(
      page.getByRole('heading', { name: /vin ocr recognition system/i })
    ).toBeVisible({ timeout: 15000 });

    await expect(
      page.getByText(/single image recognition/i, { exact: false })
    ).toBeVisible();

    await expect(
      page.getByText(/batch processing/i, { exact: false })
    ).toBeVisible();
  });
});