import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.app_logo import add_logo
from streamlit_extras.let_it_rain import rain
from streamlit_gsheets import GSheetsConnection

THIS_DIR = Path(__file__).parent
ASSETS = THIS_DIR / "assets"
LOGO = ASSETS / "savila_games_logo.png"
print(LOGO)

def verify_precedence_constraints(PROJECT,Schedule,verbose=False):

  prec_constraints = []
  invalids_list = []
  valid = True
  counter = 0
  for a in PROJECT:
    if a == 'Start':
      continue
    start = Schedule[a]['ES']
    prec = PROJECT[a]['precedence']
    #condition = []
    for c in prec:
      if verbose:
        print(f"* Precedence {c,a} | f_{c}:{Schedule[c]['EF']} <= s_{a}:{start}, {Schedule[c]['EF'] <=  start}")
      cond = Schedule[c]['EF'] <=  start
      if not cond:
        valid = False
        invalids_list.append(counter)
      prec_constraints.append({'activity':a,'predecessor':c,'start_activity':start,'finish_predecessor':Schedule[c]['EF'],'valid':cond})
      counter+=1
    #prec_constraints.append(condition)
  return prec_constraints,valid,invalids_list

def verify_resource_constraints(PROJECT,Schedule,max_resources,verbose=False):
  
  if not isinstance(max_resources,list):
    raise TypeError('max resources must be a list')
  n_resources = len(PROJECT['Start']['resources'])
  if n_resources != len(max_resources):
    raise ValueError('Then lenght of the maximun resource capacity list must be equal to the lenght of resources in the project')

  #print(n_resources)
  makespan = Schedule['End']['ES']
  #print(makespan)

  resource_consumption_schedule = np.zeros((n_resources,makespan),dtype = int)
  #print(resources_per_day.shape)

  resource_conflicts_output = []

  for day in range(0,makespan):
    if verbose:
      print(f'Day: {day}')
    for r in range(n_resources):
      C = max_resources[r]
      resources_consumed_day = 0
      potential_activity_conflicts = []
      if verbose:
        print(f' Resource: {r}')
      for a in Schedule:
        if a == 'Start' or a =='End':
          continue
        start_a = Schedule[a]['ES']
        finish_a = Schedule[a]['EF']
        if (day >= start_a) and (day < finish_a):
          resources_activity = PROJECT[a]['resources'][r]
          if resources_activity == 0:
            continue
          potential_activity_conflicts.append(a)
          if verbose:
            print(f'  Activity {a}, ({start_a},{finish_a}), u[r]:{resources_activity}')
          resources_consumed_day += resources_activity
      resource_consumption_schedule[r,day] = resources_consumed_day
      if resources_consumed_day > C:
        resource_conflicts_output.append({'activities':potential_activity_conflicts,
                                          'day':day,
                                          'resource':r,
                                          'resource_consumed':resources_consumed_day})
      if verbose:
        print(f'  Total of resource_{r} consumed on day ({day}): {resources_consumed_day}')
    if verbose:
      print('-'*120)
        
  if verbose:
    print(resource_consumption_schedule)


  return resource_consumption_schedule,resource_conflicts_output

def transform_to_manhattan(x, y):
    """Transforms data points to have a 'Manhattan plot' style."""
    new_x = []
    new_y = []
    for i in range(len(x) - 1):
        new_x.extend([x[i], x[i+1]])
        new_y.extend([y[i], y[i]])
        # Add intermediate point
        new_x.extend([x[i+1]])
        new_y.extend([y[i+1]])
    return new_x, new_y

st.set_page_config(
        page_title='AGILE PM 1666 FINAL PROJECT', # agregamos el nombre de la pagina, este se ve en el browser
        page_icon='📅' # se agrega el favicon, tambien este se ve en el browser
    )

if 'user_name' not in st.session_state:
    st.session_state['user_name'] = ''

if 'password' not in st.session_state:
    st.session_state['password'] = ''

if 'group' not in st.session_state:
    st.session_state['group'] = ''

if 'Schedule' not in st.session_state:
    st.session_state['Schedule'] = ''

if 'valid' not in st.session_state:
    st.session_state['valid'] = ''

if 'resource_schedule' not in st.session_state:
    st.session_state['resource_schedule'] = ''


df_n_cols = 34
    
PROJECT = {'Start': {'idx': 0,
  'description': 'Start of Project',
  'duration': 0,
  'precedence': None,
  'resources': [0, 0, 0, 0],
  'cost': 0},
 'A': {'idx': 1,
  'description': 'activity_A',
  'duration': 9,
  'precedence': ['Start'],
  'resources': [0, 1, 0, 0],
  'cost': 0},
 'B': {'idx': 2,
  'description': 'activity_B',
  'duration': 2,
  'precedence': ['Start'],
  'resources': [0, 1, 0, 1],
  'cost': 0},
 'C': {'idx': 3,
  'description': 'activity_C',
  'duration': 9,
  'precedence': ['Start'],
  'resources': [1, 1, 1, 1],
  'cost': 0},
 'D': {'idx': 4,
  'description': 'activity_D',
  'duration': 5,
  'precedence': ['B'],
  'resources': [0, 0, 1, 0],
  'cost': 0},
 'E': {'idx': 5,
  'description': 'activity_E',
  'duration': 7,
  'precedence': ['Start'],
  'resources': [1, 0, 0, 1],
  'cost': 0},
 'F': {'idx': 6,
  'description': 'activity_F',
  'duration': 2,
  'precedence': ['Start'],
  'resources': [0, 1, 0, 1],
  'cost': 0},
 'G': {'idx': 7,
  'description': 'activity_G',
  'duration': 8,
  'precedence': ['Start'],
  'resources': [1, 0, 0, 0],
  'cost': 0},
 'H': {'idx': 8,
  'description': 'activity_H',
  'duration': 3,
  'precedence': ['Start'],
  'resources': [0, 0, 1, 1],
  'cost': 0},
 'I': {'idx': 9,
  'description': 'activity_I',
  'duration': 6,
  'precedence': ['Start'],
  'resources': [1, 1, 1, 0],
  'cost': 0},
 'J': {'idx': 10,
  'description': 'activity_J',
  'duration': 9,
  'precedence': ['Start'],
  'resources': [1, 0, 0, 0],
  'cost': 0},
 'K': {'idx': 11,
  'description': 'activity_K',
  'duration': 1,
  'precedence': ['A', 'B', 'F'],
  'resources': [1, 0, 0, 1],
  'cost': 0},
 'L': {'idx': 12,
  'description': 'activity_L',
  'duration': 6,
  'precedence': ['A', 'C', 'E'],
  'resources': [0, 0, 1, 0],
  'cost': 0},
 'M': {'idx': 13,
  'description': 'activity_M',
  'duration': 1,
  'precedence': ['A'],
  'resources': [1, 0, 0, 1],
  'cost': 0},
 'N': {'idx': 14,
  'description': 'activity_N',
  'duration': 3,
  'precedence': ['Start'],
  'resources': [1, 1, 1, 1],
  'cost': 0},
 'O': {'idx': 15,
  'description': 'activity_O',
  'duration': 7,
  'precedence': ['C', 'D', 'E', 'G', 'K'],
  'resources': [0, 0, 0, 1],
  'cost': 0},
 'P': {'idx': 16,
  'description': 'activity_P',
  'duration': 4,
  'precedence': ['C', 'D', 'E', 'G', 'H', 'K'],
  'resources': [0, 0, 1, 1],
  'cost': 0},
 'Q': {'idx': 17,
  'description': 'activity_Q',
  'duration': 10,
  'precedence': ['D', 'F', 'L', 'M', 'N'],
  'resources': [0, 0, 1, 0],
  'cost': 0},
 'R': {'idx': 18,
  'description': 'activity_R',
  'duration': 2,
  'precedence': ['A', 'C', 'D', 'E', 'G'],
  'resources': [1, 0, 1, 1],
  'cost': 0},
 'S': {'idx': 19,
  'description': 'activity_S',
  'duration': 5,
  'precedence': ['A', 'B', 'C', 'G', 'I'],
  'resources': [0, 0, 1, 1],
  'cost': 0},
 'T': {'idx': 20,
  'description': 'activity_T',
  'duration': 7,
  'precedence': ['H', 'J', 'O', 'Q', 'S'],
  'resources': [0, 1, 0, 0],
  'cost': 0},
 'U': {'idx': 21,
  'description': 'activity_U',
  'duration': 7,
  'precedence': ['I', 'J', 'L', 'N', 'P', 'R'],
  'resources': [1, 0, 0, 1],
  'cost': 0},
 'V': {'idx': 22,
  'description': 'activity_V',
  'duration': 5,
  'precedence': ['H', 'I', 'L', 'M', 'N', 'O'],
  'resources': [0, 1, 0, 0],
  'cost': 0},
 'W': {'idx': 23,
  'description': 'activity_W',
  'duration': 9,
  'precedence': ['H', 'I', 'J', 'M', 'O'],
  'resources': [1, 1, 1, 0],
  'cost': 0},
 'X': {'idx': 24,
  'description': 'activity_X',
  'duration': 7,
  'precedence': ['H', 'I', 'J', 'L', 'O'],
  'resources': [1, 1, 0, 0],
  'cost': 0},
 'Y': {'idx': 25,
  'description': 'activity_Y',
  'duration': 5,
  'precedence': ['C', 'D', 'F', 'G', 'H', 'I', 'J', 'M', 'N'],
  'resources': [1, 1, 1, 1],
  'cost': 0},
 'Z': {'idx': 26,
  'description': 'activity_Z',
  'duration': 8,
  'precedence': ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'J'],
  'resources': [1, 0, 1, 0],
  'cost': 0},
 'AA': {'idx': 27,
  'description': 'activity_AA',
  'duration': 1,
  'precedence': ['A', 'C', 'D', 'E', 'F', 'H', 'I'],
  'resources': [1, 1, 1, 1],
  'cost': 0},
 'AB': {'idx': 28,
  'description': 'activity_AB',
  'duration': 2,
  'precedence': ['A', 'B', 'C', 'H', 'I', 'J'],
  'resources': [0, 0, 0, 1],
  'cost': 0},
 'End': {'idx': 29,
  'description': 'End of Project',
  'duration': 0,
  'precedence': ['T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB'],
  'resources': [0, 0, 0, 0],
  'cost': 0}}

max_resources = [1,1,1,1]
resources_names = [f'Resource_{x+1}' for x in range(len(max_resources))]

activities = list(PROJECT.keys())
login = None

# Establishing a Google Sheets connection
conn = st.connection("gsheets", type=GSheetsConnection)
users_existing_data =  conn.read(worksheet="users", usecols=list(range(5)), ttl=1)
users_existing_data = users_existing_data.dropna(how="all")
users_existing_data.index = users_existing_data['email']

gs_user_db = users_existing_data.T.to_dict()

#st.write(gs_user_db)

with st.sidebar:
    st.image("./savila_games_logo.png")
    selected = option_menu(
        menu_title='AGILE PM 1666',
        options= ['Login','Ranking','My group Submissions','Submit New Solution'],
        icons=['bi bi-person-fill-lock', '123','bi bi-database','bi bi-bar-chart-steps'], menu_icon="cast"
    )
    
    add_vertical_space(1)

    if selected == 'Login':

        user_name = st.text_input('User email', placeholder = 'username@ieseg.fr')
    #st.caption('Please enter your IESEG email address')
        password =  st.text_input('Password', placeholder = '12345678',type="password")
        login = st.button("Login", type="primary")
    add_vertical_space(1)

if selected == 'Login':
   
   st.header('Welcome to your Final group project Scheduling Challenge')
   st.write('Please proceed to login using the left side panel')
   st.write('⚠️ Please note that access to subsequent sections is restricted to logged-in users.')
   st.divider()
   st.subheader('Rules:')
   with st.expander('Rules',expanded=True):
      st.markdown("- This app lets you submit your project schedules.")
      st.markdown("- Everyone on your team can submit as many as they like.")
      st.markdown("- Whichever team comes up with the schedule that finishes the project fastest wins the challenge and gets the top grade.")
      st.markdown("- In case there is a tie. The team that submitted first the solution will get the higher grade.")  
      st.markdown("- Only schedules that respect the project constraints are counted towards the ranking. A valid schedule means that it respects the precedence and resource constraints. ")         
                
   st.subheader('How to use the app:')
   with st.expander('How to use the app',expanded=True):
      st.markdown("- Check out the **Ranking** 🥇 tab to see where you stand.")   
      st.markdown("- Click on **My Group Submissions** 🗃️ to see all the schedules your team has submitted.")
      st.markdown("- Hit **Submit New Solution** 📊 to drop in a new schedule anytime.")
      st.markdown("- Every time you submit, the app checks if it's all good and pops out a Gantt chart and resource plot to help you make your next schedule even better.") 
      st.markdown("- Keep trying new things and submitting - there's no limit")          
   add_vertical_space(1)
   st.image('https://media1.tenor.com/m/YoFWnXe4V3kAAAAC/may-the-odds-be-ever-in-your-favor-may-the-odds-hunger-games.gif',
            use_column_width = 'always' )

if login:
    if user_name not in gs_user_db.keys():
        st.error('Username not registered')
    else:
        real_password = gs_user_db[user_name]['password']
        if password.lower() != real_password:
            st.error('Sorry wrong password')
        else:
            user_first_name = gs_user_db[user_name]['name']
            group = gs_user_db[user_name]['group']
            st.session_state['user_name'] = user_name
            st.session_state['password'] = real_password
            st.session_state['group'] =  group
            st.success(f'{user_first_name} from group ({group}) succesfully log-in', icon="✅")

with st.sidebar:
    if st.session_state['user_name'] != '':
        st.write(f"User: {st.session_state['user_name']} ")
        st.write(f"Group: {st.session_state['group']} ")
        logout = st.button('Logout')
        if logout:
            st.session_state['user_name'] = ''
            st.session_state['password'] = ''
            st.session_state['group'] = ''
            st.session_state['Schedule'] = ''
            st.session_state['valid'] = ''
            st.session_state['resource_schedule'] = ''
            st.rerun()
    else:
        st.write(f"User: Not logged in ")


if selected == 'Submit New Solution':
    st.header('Submit New solution')
    
    if st.session_state['user_name'] == '':
        st.warning('Please log in to be able to submit your project solution')
    else:

       
        #st.dataframe(logs_df)
        #st.subheader('Submit your project schedule')
        st.write('This section of the site allows you to submit a new schedule. Please input the start date of each one of the project activities')
        with st.form(key='columns_in_form'):
            st.markdown("**Please input the start date of each project activity:**")
            n_columns = 4
            columns_form = st.columns(n_columns)
            form_dict = dict()
            for i,activity in enumerate(activities):
                col = i % n_columns
                with columns_form[col]:
                    form_dict[activity] = st.number_input(f"{activity} *", value=0,min_value=0,step=1, placeholder="0")
            st.caption("*Start* and *End* are both \'\'dummy\'\' activities. For the *End* activity please input the final date of your project.")
            st.markdown("**required*")
            submitButton = st.form_submit_button(label = 'Submit',type='primary')
            if submitButton:

                timestamp = datetime.datetime.now()
                timestamp = timestamp.strftime("%d/%m/%Y, %H:%M:%S")
                st.write(timestamp)
                solution_dict = dict()
                solution_dict['user'] = st.session_state['user_name']
                solution_dict['group'] = st.session_state['group']
                solution_dict['time'] = timestamp
                Schedule = dict()
                for activity in form_dict:
                    solution_dict[activity] = form_dict[activity]
                    start = form_dict[activity]
                    finish = start + PROJECT[activity]['duration']
                    row = {'ES':start,'EF':finish}
                    Schedule[activity] = row

                st.session_state['Schedule'] = Schedule
                with st.status("Analysis of schedule...", expanded=True) as status:
                    st.write("**Verifying precedence constraints...**")
                    valid_dict,valid_bool,invalids = verify_precedence_constraints(PROJECT,Schedule,verbose=False)
                    if not valid_bool:
                        for idx in invalids:
                            inval = valid_dict[idx]
                            pred = inval['predecessor']
                            succ = inval['activity']
                            st.write(f' * ⚠️ Precedence constraint violated for ({succ},{pred})')
                    else:
                        st.write("   ✅ Precedence constraints validated")
                    st.write("**Verifying resource constraints...**")
                    resource_schedule, resource_conflicts = verify_resource_constraints(PROJECT,Schedule,max_resources,verbose=False)
                    st.session_state['resource_schedule'] = resource_schedule
                    if len(resource_conflicts) != 0:
                       for conf in resource_conflicts:
                          a_conf = conf["activities"]
                          day_conf = conf["day"]
                          res_conf = conf["resource"]
                          st.write(f' * ⚠️ Resource conflict between {a_conf}, on day ({day_conf}), for resource ({res_conf})')
                    else:
                       st.write("   ✅ Resources constraints validated ")
                    
                    if valid_bool and len(resource_conflicts) == 0:
                        solution_dict['valid'] = 'YES'
                        status.update(label="Verification completed")
                    else:
                        solution_dict['valid'] = 'NO'
                        status.update(label="Solution not valid",state="error")
                    
                with st.spinner('Uploading solution to database'):
                    logs_df = conn.read(worksheet="game_log", usecols=list(range(df_n_cols)), ttl=1).dropna(how="all")
                    solution = pd.DataFrame([solution_dict])
                    updated_log = pd.concat([logs_df,solution],ignore_index=True)
                    conn.update(worksheet="game_log",data = updated_log)
                    if solution_dict['valid'] == 'YES':
                        st.session_state['valid'] = 'YES'
                        st.success(f'Your solution was uploaded on: {timestamp}',icon="✅")
                        st.balloons()
                    else:
                        st.warning(f'Your solution was uploaded on: {timestamp} but it was not valid',icon="⚠️")
                        st.session_state['valid'] = 'NO'
        
        if st.session_state['Schedule'] != '':
            Schedule = st.session_state['Schedule']
            valid = st.session_state['valid']
            resource_schedule = st.session_state['resource_schedule']
            st.subheader('Project Gantt Chart')
            if valid == 'YES':
               gantt_color = 'royalblue'
            else:
               gantt_color = 'red'
            fig = go.Figure()
            for task in PROJECT:
                fig.add_trace(go.Bar(
                    x=[Schedule[task]["EF"] - Schedule[task]["ES"]], # Width of the bar
                    y=[task],
                    base=[Schedule[task]["ES"]], # Starting point of the bar
                    orientation='h', # Make the bars horizontal
                    marker_color=gantt_color,
                    text=task,  # Label to display inside the bar
                    textposition='inside',
                    name=task, # Use category as legend names
                    hoverinfo= None, # Show hover info
                    hovertemplate=f"Task: {task}<br>Start: {Schedule[task]['ES']}<br>Finish: {Schedule[task]['EF']}<extra></extra>",  # Custom hover text
                    ))

            # Customize layout
            fig.update_layout(
                title=f"Project Final duration = {Schedule['End']['EF']} days",
                xaxis_title="Days",
                yaxis_title="Project Activities",
                barmode='stack', # Stack bars to create the broken bar effect
                bargap=0.1, # Gap between bars of the same y level
            )
            fig.update_layout(yaxis_autorange='reversed')
            fig.update_layout(showlegend=False)
            fig.update_layout(
                xaxis=dict(
                    dtick=5  # Adjust this value as needed for your data range
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader('Resource Level plots')

            resource_tabs = st.tabs(resources_names)
            resource_color = ['blue','red','orange','limegreen','grey']
            for r,res_tab in enumerate(resource_tabs):
               
               with res_tab:
                    fig2 = go.Figure()
                    row = resource_schedule[r]
                    x_data = np.arange(len(row))
                    y_data = row
                    new_x, new_y = transform_to_manhattan(x_data, y_data)
                    last_x = new_x[-1] +1
                    last_y = new_y[-1]
                    new_x.append(last_x)
                    new_y.append(last_y)
                    fig2.add_trace(go.Scatter(
                        x=new_x,  # X-axis: range from 0 to the length of the row
                        y=new_y,  # Y-axis: row values
                        line=dict(width=3,color = resource_color[r]),
                        mode='lines+markers',  # Plot lines and markers
                        name=f'Resource {r+1}'  # Name of the line (for legend)
                    ))

                    # Update layout
                    fig2.update_layout(
                        title=f'Resource {r+1}',
                        xaxis_title="Days",
                        yaxis_title="Resource Consumption",
                        legend_title=None
                    )

                    # Show figure
                    # Add an extra horizontal line at y=1 with a label
                    x_range = np.arange(resource_schedule.shape[1]+1)

                    # Add a horizontal line as a scatter trace
                    fig2.add_trace(go.Scatter(x=x_range, y=np.ones_like(x_range), 
                                            mode='lines', 
                                            line=dict(color='black', width=3,dash='dash'),
                                            name='max_capacity',
                                            hoverinfo='skip'))
                    fig2.update_layout(
                        xaxis=dict(
                            dtick=5  # Adjust this value as needed for your data range
                        )
                    )
                    fig2.update_layout(
                        yaxis=dict(
                            dtick=1  # Adjust this value as needed for your data range
                        )
                    )
                    fig2.update_layout(
                        xaxis=dict(
                            range=[0, max(x_range)+1]  # Set x-axis to start at 0 and end at the max value of x_range
                        )
                    )
                    fig2.update_layout(
                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=-0.5,  # Adjusts the vertical position of the legend
                            xanchor='center',
                            x=0.5  # Centers the legend horizontally
                        )
                    )
                    st.plotly_chart(fig2, use_container_width=True)

if selected == "Ranking":
    st.header('Ranking')
    
    if st.session_state['user_name'] == '':
        st.warning('Please log in to be able to submit your project solution')
    else:
        st.write('The table below shows the competition ranking')
        submissions_log_df = conn.read(worksheet="game_log", usecols=list(range(df_n_cols)), ttl=1).dropna(how="all")
        default_time = pd.to_datetime('01/01/1901, 00:00:00',format="%d/%m/%Y, %H:%M:%S")
        ranking_list = []
        for gr in ['G1','G2','G3','G4','PR']:
            mini_df = submissions_log_df[(submissions_log_df['group'] == gr) & (submissions_log_df['valid'] == 'YES')]
            if len(mini_df) == 0:
                row = {'group':gr,'Project Duration':1_000,'time':default_time}
                ranking_list.append(row)
                continue
            else:
                best_idx = np.argmin(mini_df['End'])
                best_value = mini_df.iat[best_idx,-2]
                best_time = pd.to_datetime(mini_df.iat[best_idx,2],format="%d/%m/%Y, %H:%M:%S")
                row = {'group':gr,'Project Duration':best_value,'time':best_time}
                ranking_list.append(row)

        ranking_df = pd.DataFrame(ranking_list).sort_values(by = ['Project Duration','time'])
        ranking_df = ranking_df.reset_index(drop=True)
        ranking_df.iat[0,0] = ranking_df.iat[0,0] + "   🥇"
        ranking_df.iat[1,0] = ranking_df.iat[1,0] + "   🥈"
        ranking_df.iat[2,0] = ranking_df.iat[2,0] + "   🥉"
        st.dataframe(ranking_df,use_container_width=True,hide_index=True)
        st.caption('Remember that only valid solutions classify for the ranking. Schedules with the lowest duration go to the top of the ranking. In case of a tie, the position is decided based on the date-hour of the submission. Whoever submitted the schedule first wins in this case.')
        

if selected == 'My group Submissions':
    st.header('My Group Submissions')
    
    if st.session_state['user_name'] == '':
        st.warning('Please log in to be able to submit your project solution')
    else:
        st.write(f'The table below shows you the submission history of your group: **{st.session_state["group"]}**')
        group_log_df = conn.read(worksheet="game_log", usecols=list(range(df_n_cols)), ttl=1).dropna(how="all")
        group_log_df = group_log_df[group_log_df['group'] == st.session_state['group']]
        group_log_df = group_log_df[['user','time','End','valid']]
        group_log_df = group_log_df.rename(columns = {'End':'Project Duration'})
       
        st.subheader('Submissions History:')
        st.dataframe(group_log_df,use_container_width=True,hide_index=True)
