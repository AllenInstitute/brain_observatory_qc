# mindscope_qc
This is a central location for tracking data quality issues related to ongoing Mindscope projects.

Anyone with data quality concerns should log issues here to ensure that they are tracked and followed up on. This repository should be a central place to log and track issues, along with any associated documents (Jupyter Notebooks, PDF writeups, slide decks, etc.) that were prepared in the process of tracking or following up on issues.

This is not intended to be a code repository.

## Creating Issues
If you note a new data issue, here are steps to follow to log it:
1. Click on the 'Issues' tab to the top/left of the page:
![image](https://user-images.githubusercontent.com/19944442/128929021-1cde3fab-414e-4e92-bca3-f5d16b79007c.png)

2. Try searching for related issues. If there is a currently an open issue that covers the same topic, consider adding to that issue rather than creating a new issue.

3. If a new issue is warranted, use the button to create a new issue and select the appropriate issue type: 
   * bug report: report bugs in mouse seeks or the current qc system. Examples include plots not populating, a metric no longer functioning as expected 
   * QC issue: if there is a new qc issue or trend that arises that isn't covered by the current qc reporting system. Examples include consistent failures on a particular microscope, discovery of some previously unknown failure modality
   * feature request: to add new metrics or plots or update existing metrics & plots
   
4. Be as descriptive as possible with your issue reporting. Use the issue template as a guide but please include:
    * A descriptive title
    * Text describing the context - how did you discover the issue, what do you think might be causing it, etc. Be as descriptive as possible!
    * Screenshots, photos, etc. Github makes it very easy to include images inline with text.
    * Relevant code blocks. If you discovered the issue during analysis, please include all code necessary to re-create the issue. Raw text is preferrable to a screenshot of code - this allows someone following up to copy/paste/run your code. Use the 'insert code' formatting button to delineate code from text
    ![image](https://user-images.githubusercontent.com/19944442/128932459-39f3ad8e-3d0d-46d3-96d5-7f9a226175a3.png)
    * Links to any other relevant issues. If you are referencing other issues in this repository, simply typing `#` followed by the issue number will create a hyperlink
    
5. Add relevant labels. The 'Labels' dropdown is just to the right of the main comment field. Consider if your issue or feature request is related to a specific:
   * project
   * microscope or microscope type
   * data stream (e.g. eye tracking, running wheel etc.)
  
6. Tag any relevant individuals in the issue text. Including their github username after the `@` symbol will ensure that they are notified of the issue and any followup comments.

7. Assign someone, including yourself. The individual(s) listed under `Assignees` are responsible for followup. Add people if you know they should be here, or leave it blank if you're not sure.

8. Follow up! If you have additional information or context to provide later, return to the issue and add new comments. Try to avoid editing previous comments as this makes tracking difficult.

9. Close the issue when it is solved.
