import { test, expect } from '@playwright/test';

test('searches for Garfield and verifies the movie is in the list', async ({ page }) => {
  await page.goto('https://debs-obrien.github.io/playwright-movies-app');

  // Click the search button to reveal the input, forcing the click in case of pointer interception
  await page.getByRole('button', { name: 'Search for a movie' }).click({ force: true });

  // Now the input should be visible
  await page.getByPlaceholder('Search for a movie...').fill('Garfield');

  // Wait for the results to update (could also wait for a specific result)
  await page.waitForTimeout(500);

  // Verify 'Garfield' is in the movie list
  await expect(page.getByText('Garfield')).toBeVisible();
});