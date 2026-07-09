# Task Template Augmentations for SuiteCRM (Templates 2000-2008)

This document outlines 5 augmentation options for each task template. Each augmentation represents a different field combination while maintaining the same general task type.

---

## Template 2000: Create Account

### Current: Name only
**Intent Template:** `"Create a new account with the name '{{account_name}}'"`

### Augmentation Options:

#### Option 1: Account with Phone
- **Intent Template:** `"Create a new account with the name '{{account_name}}' and office phone '{{phone}}'"`
- **Fields:** Name, Office Phone
- **Evaluation:** Check for account name and phone number in saved record

#### Option 2: Account with Description
- **Intent Template:** `"Create a new account with the name '{{account_name}}' with description '{{description}}'"`
- **Fields:** Name, Description
- **Evaluation:** Check for account name and description

#### Option 3: Account with Location
- **Intent Template:** `"Create a new account with the name '{{account_name}}' located in '{{city}}', '{{country}}'"`
- **Fields:** Name, Billing City, Billing Country
- **Evaluation:** Check for account name, city, and country

#### Option 4: Account with Website & Description
- **Intent Template:** `"Create a new account with the name '{{account_name}}' with website '{{website}}' and description '{{description}}'"`
- **Fields:** Name, Website, Description
- **Evaluation:** Check for account name, website URL, and description

#### Option 5: Account with Website & Email
- **Intent Template:** `"Create a new account with the name '{{account_name}}' with website '{{website}}' and email '{{email}}'"`
- **Fields:** Name, Website, Email Address
- **Evaluation:** Check for account name, website URL, and email

---

## Template 2001: Create Contact

### Current: Full Name + Job Title
**Intent Template:** `"Create a new contact with the name '{{full_name}}' and job title '{{job_title}}'"`

### Augmentation Options:

#### Option 1: Contact with Phone
- **Intent Template:** `"Create a new contact with the name '{{full_name}}' and office phone '{{phone}}'"`
- **Fields:** First Name, Last Name, Office Phone
- **Evaluation:** Check for full name and phone number

#### Option 2: Contact with Email
- **Intent Template:** `"Create a new contact with the name '{{full_name}}' and email '{{email}}'"`
- **Fields:** First Name, Last Name, Email Address
- **Evaluation:** Check for full name and email

#### Option 3: Contact with Department
- **Intent Template:** `"Create a new contact with the name '{{full_name}}' in the '{{department}}' department"`
- **Fields:** First Name, Last Name, Department
- **Evaluation:** Check for full name and department

#### Option 4: Contact with Account
- **Intent Template:** `"Create a new contact with the name '{{full_name}}' associated with account '{{account_name}}'"`
- **Fields:** First Name, Last Name, Account Name (relation dropdown)
- **Evaluation:** Check for full name and linked account

#### Option 5: Contact with Salutation & Mobile
- **Intent Template:** `"Create a new contact '{{salutation}} {{full_name}}' with mobile phone '{{mobile}}'"`
- **Fields:** Salutation (Mr./Ms./Dr./Prof.), First Name, Last Name, Mobile
- **Evaluation:** Check for salutation, full name, and mobile

---

## Template 2002: Create Lead

### Current: Last Name + Email
**Intent Template:** `"Create a new lead with last name '{{last_name}}' and email '{{email}}'"`

### Augmentation Options:

#### Option 1: Lead with Phone
- **Intent Template:** `"Create a new lead with last name '{{last_name}}' and office phone '{{phone}}'"`
- **Fields:** Last Name, Office Phone
- **Evaluation:** Check for last name and phone

#### Option 2: Lead with Company
- **Intent Template:** `"Create a new lead with last name '{{last_name}}' from company '{{account_name}}' with website '{{website}}'"`
- **Fields:** Last Name, Account Name (company), Website
- **Evaluation:** Check for last name, company name, and website

#### Option 3: Lead with Status
- **Intent Template:** `"Create a new lead with last name '{{last_name}}' and email '{{email}}' with status '{{status}}'"`
- **Fields:** Last Name, Email, Status (dropdown: New, Assigned, In Process)
- **Evaluation:** Check for last name, email, and status

#### Option 4: Lead with Department
- **Intent Template:** `"Create a new lead with last name '{{last_name}}' in the '{{department}}' department"`
- **Fields:** Last Name, Department
- **Evaluation:** Check for last name and department

#### Option 5: Lead with Location
- **Intent Template:** `"Create a new lead with last name '{{last_name}}' located in '{{city}}', '{{country}}'"`
- **Fields:** Last Name, Primary Address City, Primary Address Country
- **Evaluation:** Check for last name, city, and country

---

## Template 2003: Create Opportunity

### Current: Name + Amount + Account + Close Date + Sales Stage
**Intent Template:** `"Create a new opportunity '{{opportunity_name}}' for account '{{account_name}}' with amount '{{amount}}', close date '{{close_date}}', and sales stage '{{sales_stage}}'"`

### Augmentation Options:

#### Option 1: Opportunity with Type
- **Intent Template:** `"Create a new opportunity '{{opportunity_name}}' for account '{{account_name}}' with amount '{{amount}}', close date '{{close_date}}', sales stage '{{sales_stage}}', and type '{{type}}'"`
- **Fields:** Opportunity Name, Account Name, Amount, Close Date, Sales Stage, Type (Existing Business/New Business)
- **Evaluation:** Check for all fields including type

#### Option 2: Opportunity with Probability
- **Intent Template:** `"Create a new opportunity '{{opportunity_name}}' for account '{{account_name}}' with amount '{{amount}}', close date '{{close_date}}', sales stage '{{sales_stage}}', and probability '{{probability}}%'"`
- **Fields:** Opportunity Name, Account Name, Amount, Close Date, Sales Stage, Probability (%)
- **Evaluation:** Check for all fields including probability

#### Option 3: Opportunity with Next Step
- **Intent Template:** `"Create a new opportunity '{{opportunity_name}}' for account '{{account_name}}' with close date '{{close_date}}', sales stage '{{sales_stage}}', and next step '{{next_step}}'"`
- **Fields:** Opportunity Name, Account Name, Close Date, Sales Stage, Next Step
- **Evaluation:** Check for all fields including next step

#### Option 4: Opportunity with Lead Source
- **Intent Template:** `"Create a new opportunity '{{opportunity_name}}' for account '{{account_name}}' with amount '{{amount}}', close date '{{close_date}}', and lead source '{{lead_source}}'"`
- **Fields:** Opportunity Name, Account Name, Amount, Close Date, Lead Source (dropdown)
- **Evaluation:** Check for all fields including lead source

#### Option 5: Opportunity with Description
- **Intent Template:** `"Create a new opportunity '{{opportunity_name}}' for account '{{account_name}}' with amount '{{amount}}', close date '{{close_date}}', sales stage '{{sales_stage}}', and description '{{description}}'"`
- **Fields:** Opportunity Name, Account Name, Amount, Close Date, Sales Stage, Description
- **Evaluation:** Check for all fields including description

---

## Template 2004: Create Task

### Current: Subject + Start Date + Priority
**Intent Template:** `"Create a new task with subject '{{subject}}', start date '{{start_date}}', and priority '{{priority}}'"`

### Augmentation Options:

#### Option 1: Task with Due Date
- **Intent Template:** `"Create a new task with subject '{{subject}}', due date '{{due_date}}', and priority '{{priority}}'"`
- **Fields:** Subject, Due Date, Priority
- **Evaluation:** Check for subject, due date, and priority

#### Option 2: Task with Status
- **Intent Template:** `"Create a new task with subject '{{subject}}', start date '{{start_date}}', and status '{{status}}'"`
- **Fields:** Subject, Start Date, Status (Not Started/In Progress/Completed/Pending Input/Deferred)
- **Evaluation:** Check for subject, start date, and status

#### Option 3: Task with Description
- **Intent Template:** `"Create a new task with subject '{{subject}}', start date '{{start_date}}', priority '{{priority}}', and description '{{description}}'"`
- **Fields:** Subject, Start Date, Priority, Description
- **Evaluation:** Check for subject, start date, priority, and description

---

## Template 2005: Update Contact

### Current: Update Mobile Number
**Intent Template:** `"Go to the Contact '{{contact_name}}' and update the mobile number to '{{mobile}}'"`

### Augmentation Options:

#### Option 1: Update Job Title
- **Intent Template:** `"Go to the Contact '{{contact_name}}' and update the job title to '{{job_title}}'"`
- **Fields:** Find contact, Update Job Title
- **Evaluation:** Check that job title is updated in saved record

#### Option 2: Update Department
- **Intent Template:** `"Go to the Contact '{{contact_name}}' and update the department to '{{department}}'"`
- **Fields:** Find contact, Update Department
- **Evaluation:** Check that department is updated

#### Option 3: Update Office Phone
- **Intent Template:** `"Go to the Contact '{{contact_name}}' and update the office phone to '{{phone}}'"`
- **Fields:** Find contact, Update Office Phone
- **Evaluation:** Check that office phone is updated

#### Option 4: Update Email
- **Intent Template:** `"Go to the Contact '{{contact_name}}' and update the email address to '{{email}}'"`
- **Fields:** Find contact, Update Email Address
- **Evaluation:** Check that email is updated

#### Option 5: Update Address
- **Intent Template:** `"Go to the Contact '{{contact_name}}' and update the primary address to '{{city}}', '{{country}}'"`
- **Fields:** Find contact, Update Primary Address City, Update Primary Address Country
- **Evaluation:** Check that address fields are updated

---

## Template 2006: Update Opportunity

### Current: Update Close Date + Next Step
**Intent Template:** `"Go to the Opportunity '{{opportunity_name}}' and update the close date to '{{close_date}}' and next step to '{{next_step}}'"`

### Augmentation Options:

#### Option 1: Update Amount
- **Intent Template:** `"Go to the Opportunity '{{opportunity_name}}' and update the opportunity amount to '{{amount}}'"`
- **Fields:** Find opportunity, Update Opportunity Amount
- **Evaluation:** Check that amount is updated

#### Option 2: Update Sales Stage
- **Intent Template:** `"Go to the Opportunity '{{opportunity_name}}' and update the sales stage to '{{sales_stage}}'"`
- **Fields:** Find opportunity, Update Sales Stage (dropdown)
- **Evaluation:** Check that sales stage is updated

#### Option 3: Update Probability
- **Intent Template:** `"Go to the Opportunity '{{opportunity_name}}' and update the probability to '{{probability}}%'"`
- **Fields:** Find opportunity, Update Probability (%)
- **Evaluation:** Check that probability is updated

#### Option 4: Update Type
- **Intent Template:** `"Go to the Opportunity '{{opportunity_name}}' and update the type to '{{type}}'"`
- **Fields:** Find opportunity, Update Type (Existing Business/New Business)
- **Evaluation:** Check that type is updated

#### Option 5: Update Description
- **Intent Template:** `"Go to the Opportunity '{{opportunity_name}}' and update the description to '{{description}}'"`
- **Fields:** Find opportunity, Update Description
- **Evaluation:** Check that description is updated

---

## Template 2007: Update Lead

### Current: Update Department
**Intent Template:** `"Go to the Lead '{{lead_name}}' and update the department to '{{department}}'"`

### Augmentation Options:

#### Option 1: Update Status
- **Intent Template:** `"Go to the Lead '{{lead_name}}' and update the status to '{{status}}'"`
- **Fields:** Find lead, Update Status (New/Assigned/In Process/Converted/Recycled/Dead)
- **Evaluation:** Check that status is updated

#### Option 2: Update Mobile
- **Intent Template:** `"Go to the Lead '{{lead_name}}' and update the mobile number to '{{mobile}}'"`
- **Fields:** Find lead, Update Mobile Number
- **Evaluation:** Check that mobile is updated

#### Option 3: Update Department
- **Intent Template:** `"Go to the Lead '{{lead_name}}' and update the department to '{{department}}'"`
- **Fields:** Find lead, Update Department
- **Evaluation:** Check that department is updated

#### Option 4: Update Website
- **Intent Template:** `"Go to the Lead '{{lead_name}}' and update the website to '{{website}}'"`
- **Fields:** Find lead, Update Website
- **Evaluation:** Check that website is updated

#### Option 5: Update Job Title
- **Intent Template:** `"Go to the Lead '{{lead_name}}' and update the job title to '{{job_title}}'"`
- **Fields:** Find lead, Update Job Title
- **Evaluation:** Check that job title is updated

---

## Template 2008: Update Account

### Current: Update Office Phone
**Intent Template:** `"Go to the Account '{{account_name}}' and update the office phone to '{{phone}}'"`

### Augmentation Options:

#### Option 1: Update Website
- **Intent Template:** `"Go to the Account '{{account_name}}' and update the website to '{{website}}'"`
- **Fields:** Find account, Update Website
- **Evaluation:** Check that website is updated

#### Option 2: Update Description
- **Intent Template:** `"Go to the Account '{{account_name}}' and update the description to '{{description}}'"`
- **Fields:** Find account, Update Description
- **Evaluation:** Check that description is updated

#### Option 3: Update Website
- **Intent Template:** `"Go to the Account '{{account_name}}' and update the website to '{{website}}'"`
- **Fields:** Find account, Update Website
- **Evaluation:** Check that website is updated

#### Option 4: Update Annual Revenue
- **Intent Template:** `"Go to the Account '{{account_name}}' and update the annual revenue to '{{revenue}}'"`
- **Fields:** Find account, Update Annual Revenue
- **Evaluation:** Check that annual revenue is updated

#### Option 5: Update Employees
- **Intent Template:** `"Go to the Account '{{account_name}}' and update the number of employees to '{{employees}}'"`
- **Fields:** Find account, Update Employees
- **Evaluation:** Check that employees count is updated

---

## Notes

- All augmentations maintain the same policy structure as the base templates
- Evaluation uses `program_html` with `document.body.innerText` for update tasks (to check saved values)
- For update tasks, the `start_url` may need to be a record page or list page depending on the task
- Field names in the intent should match the actual form field labels in SuiteCRM
- Dropdown values should match the exact options available in the SuiteCRM forms
